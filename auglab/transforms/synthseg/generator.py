"""SynthSeg generative model (``BrainGenerator``) as a torch module.

``SynthSegGenerator`` faithfully reproduces the order of operations and the
default hyper-parameters of the SynthSeg "brain generator"
(``SynthSeg/brain_generator.py`` -> ``labels_to_image_model``), turning an
anatomical *label map* into a randomly-synthesised image together with the
matching (spatially-deformed, possibly relabelled) ground-truth label map:

    spatial deform (affine + diffeomorphic SVF, on labels, nearest)
      -> [optional random crop]
      -> left/right flip with label swap
      -> GMM intensity sampling (labels -> image)
      -> bias field
      -> intensity augmentation (clip -> min-max norm -> gamma)
      -> resolution randomisation (blur -> subsample -> resample), per channel
      -> map generation labels to output/segmentation labels

The default hyper-parameters match ``BrainGenerator`` (which overrides several
``labels_to_image_model`` signature defaults). See ``README.md`` for the table
and source citations.

Note on label maps: SynthSeg derives its realism from a *dense* anatomical label
map (e.g. a FreeSurfer/SAMSEG segmentation covering every tissue). When fed a
sparse target segmentation (only a few foreground structures over a 0
background) it still runs correctly, but the synthetic image will only contain
those structures over a single-Gaussian background.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from auglab.transforms.synthseg import functional as FN

Number = Union[int, float]


class SynthSegGenerator(nn.Module):
    """Differentiable-free SynthSeg image generator for 3D label maps.

    Args:
        generation_labels: the label values the GMM generates intensities for.
            If ``None`` they are inferred from each input as the sorted unique
            values (re-inferred per call).
        output_labels: label values written to the output segmentation, aligned
            with ``generation_labels`` (``generation_labels[i] -> output_labels[i]``).
            ``None`` -> identity (output labels == generation labels).
        n_neutral_labels: number of non-lateralised labels at the start of
            ``generation_labels``; enables anatomically-correct L/R label swapping
            on flip. ``None`` -> plain flip without relabelling.
        generation_classes: per-label class index so several labels can share one
            Gaussian (e.g. tie left/right). ``None`` -> every label independent.
        n_channels: number of synthesised image channels (modalities).
        prior_distributions: ``"uniform"`` or ``"normal"`` GMM parameter priors.
        prior_means / prior_stds: priors for the GMM means/stds. ``None`` ->
            ``U(0, 250)`` / ``U(0, 30)`` (full domain randomisation). May be a
            scalar, ``[a, b]``, or a ``(2, n_classes)`` array.
        flipping: enable random left/right flipping.
        scaling_bounds / rotation_bounds / shearing_bounds / translation_bounds:
            affine augmentation ranges (see :func:`functional.sample_affine_matrices`).
        nonlin_std / nonlin_scale / svf_integration_steps: diffeomorphic elastic
            deformation controls.
        bias_field_std / bias_scale: multiplicative bias-field controls.
        gamma_std / clip: intensity-augmentation controls.
        normalise: min-max normalise to [0, 1] before gamma (SynthSeg always does).
        randomise_res: randomise the acquisition resolution per channel.
        max_res_iso / max_res_aniso: resolution ceilings for ``randomise_res``.
        data_res / thickness: fixed acquisition resolution(s) when
            ``randomise_res=False`` (``(3,)`` or per-channel ``(n_channels, 3)``).
        blur_range: random jitter factor on the blur sigma (1.03 in SynthSeg).
        atlas_res: native resolution of the input label map (mm).
        output_shape: spatial size of the output (random crop of the label map
            before generation). ``None`` -> keep the input shape.
        em_label_completion: for sparse/incomplete label maps, subdivide every
            label into intensity-coherent generation sub-labels by clustering the
            paired real image with Expectation-Maximization (SynthSeg §5.4).
            Requires ``image`` to be passed to :meth:`forward`. The sub-labels are
            merged back to their parent labels in the output segmentation.
        em_n_foreground_clusters: EM sub-clusters per foreground label (2 in the
            paper).
        em_background_clusters_range: ``(min, max)`` random number of EM clusters
            for the background label ([3, 10] in the paper).
        em_background_label: label value treated as background for EM.
        em_n_iters: EM iterations.
        em_max_fit_voxels: subsample size for fitting each EM (the full region is
            still assigned); ``0`` -> use all voxels.
        em_same_on_batch: share the random background cluster count across the batch.
        apply_intensity_augmentation: toggle clip/normalise/gamma.
    """

    def __init__(
        self,
        generation_labels: Optional[Sequence[int]] = None,
        output_labels: Optional[Sequence[int]] = None,
        n_neutral_labels: Optional[int] = None,
        generation_classes: Optional[Sequence[int]] = None,
        n_channels: int = 1,
        prior_distributions: str = "uniform",
        prior_means=None,
        prior_stds=None,
        flipping: bool = True,
        flip_axis: int = 2,
        scaling_bounds: Union[bool, Number, Sequence[Number]] = 0.2,
        rotation_bounds: Union[bool, Number, Sequence[Number]] = 15.0,
        shearing_bounds: Union[bool, Number, Sequence[Number]] = 0.012,
        translation_bounds: Union[bool, Number, Sequence[Number]] = False,
        nonlin_std: float = 4.0,
        nonlin_scale: float = 0.04,
        svf_integration_steps: int = 7,
        bias_field_std: float = 0.7,
        bias_scale: float = 0.025,
        gamma_std: float = 0.5,
        clip: float = 300.0,
        normalise: bool = True,
        randomise_res: bool = True,
        max_res_iso: float = 4.0,
        max_res_aniso: float = 8.0,
        data_res=None,
        thickness=None,
        blur_range: float = 1.03,
        atlas_res: float = 1.0,
        output_shape: Optional[Sequence[int]] = None,
        em_label_completion: bool = False,
        em_n_foreground_clusters: int = 2,
        em_background_clusters_range: Sequence[int] = (3, 10),
        em_background_label: int = 0,
        em_n_iters: int = 20,
        em_max_fit_voxels: int = 100000,
        em_same_on_batch: bool = False,
        apply_affine: bool = True,
        apply_nonlinear: bool = True,
        apply_bias_field: bool = True,
        apply_intensity_augmentation: bool = True,
        apply_resolution: bool = True,
    ) -> None:
        super().__init__()
        self.generation_labels = list(generation_labels) if generation_labels is not None else None
        self.output_labels = list(output_labels) if output_labels is not None else None
        self.n_neutral_labels = n_neutral_labels
        self.generation_classes = list(generation_classes) if generation_classes is not None else None
        self.n_channels = int(n_channels)
        self.prior_distributions = prior_distributions
        self.prior_means = prior_means
        self.prior_stds = prior_stds

        self.flipping = flipping
        self.flip_axis = flip_axis
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds

        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.svf_integration_steps = int(svf_integration_steps)

        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale

        self.gamma_std = gamma_std
        self.clip = clip
        self.normalise = normalise

        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.data_res = data_res
        self.thickness = thickness
        self.blur_range = blur_range
        self.atlas_res = float(atlas_res)
        self.output_shape = list(output_shape) if output_shape is not None else None

        self.em_label_completion = em_label_completion
        self.em_n_foreground_clusters = int(em_n_foreground_clusters)
        self.em_background_clusters_range = tuple(em_background_clusters_range)
        self.em_background_label = int(em_background_label)
        self.em_n_iters = int(em_n_iters)
        self.em_max_fit_voxels = int(em_max_fit_voxels)
        self.em_same_on_batch = em_same_on_batch
        self._warned_em = False

        self.apply_affine = apply_affine
        self.apply_nonlinear = apply_nonlinear
        self.apply_bias_field = apply_bias_field
        self.apply_intensity_augmentation = apply_intensity_augmentation
        self.apply_resolution = apply_resolution

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self, label_map: torch.Tensor, image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate an image and its label map from an input label map.

        Args:
            label_map: ``(B, 1, D, H, W)`` integer / one-hot / ``(B, D, H, W)``
                segmentation. Coerced via :func:`functional.to_label_map`.
            image: ``(B, C, D, H, W)`` paired *real* image. Only used (and only
                required) when ``em_label_completion=True``, to cluster unlabelled
                tissue into generation sub-labels via EM (SynthSeg §5.4).

        Returns:
            ``(image, labels)`` where ``image`` is ``(B, n_channels, *out)`` float
            in [0, 1] (if normalised) and ``labels`` is ``(B, 1, *out)`` int.
        """
        labels = FN.to_label_map(label_map)
        device = labels.device
        batch = labels.shape[0]

        # Label bookkeeping (defaults from config; overridden by EM completion).
        gen_labels = self.generation_labels        # list[int] or None
        out_labels_cfg = self.output_labels         # list[int] or None
        gen_classes = self.generation_classes       # list[int] or None
        n_neutral = self.n_neutral_labels
        randomise_bg = True

        # 0. EM label completion for sparse maps (uses the paired real image) --
        if self.em_label_completion:
            if image is None:
                if not self._warned_em:
                    print(f"{type(self).__name__}: em_label_completion is enabled but no "
                          f"image was provided; falling back to plain generation.", flush=True)
                    self._warned_em = True
            else:
                ref = image if image.dim() == 5 else image.unsqueeze(1)
                labels, gen_labels, out_labels_cfg = FN.em_subdivide_labels(
                    ref.float(), labels,
                    n_foreground_clusters=self.em_n_foreground_clusters,
                    background_clusters_range=self.em_background_clusters_range,
                    background_label=self.em_background_label,
                    n_iters=self.em_n_iters,
                    max_fit_voxels=self.em_max_fit_voxels,
                    same_on_batch=self.em_same_on_batch,
                )
                gen_classes = None      # each sub-label gets its own Gaussian
                n_neutral = None        # plain flip (sub-labels carry no L/R structure)
                randomise_bg = False    # background is now modelled by its clusters

        # 1. random crop to output_shape (label space) ------------------------
        if self.output_shape is not None and tuple(self.output_shape) != tuple(labels.shape[2:]):
            labels = self._random_crop(labels, self.output_shape)

        # 2. spatial deformation of the LABEL MAP (nearest) -------------------
        affine = None
        if self.apply_affine and self._affine_active():
            affine = FN.sample_affine_matrices(
                batch, device,
                scaling_bounds=self.scaling_bounds,
                rotation_bounds=self.rotation_bounds,
                shearing_bounds=self.shearing_bounds,
                translation_bounds=self.translation_bounds,
            )
        displacement = None
        if self.apply_nonlinear and self.nonlin_std and self.nonlin_std > 0:
            displacement = FN.random_svf_field(
                batch, tuple(labels.shape[2:]), device,
                nonlin_std=self.nonlin_std,
                nonlin_scale=self.nonlin_scale,
                int_steps=self.svf_integration_steps,
            )
        if affine is not None or displacement is not None:
            labels = FN.warp_volume(
                labels.float(), affine=affine, displacement=displacement,
                interp="nearest", padding_mode="zeros",
            ).round().long()

        # 3. left/right flipping (with optional label swap) -------------------
        if self.flipping and float(torch.rand((), device=device)) < 0.5:
            label_values_flip = (
                torch.as_tensor(gen_labels, dtype=torch.long, device=device)
                if gen_labels is not None else FN.infer_label_values(labels)
            )
            labels = FN.flip_lr_with_swap(
                labels, self.flip_axis,
                label_values=label_values_flip,
                n_neutral_labels=n_neutral,
            )

        # The generation labels (after potential relabelling) used by the GMM.
        gen_values = (
            torch.as_tensor(gen_labels, dtype=torch.long, device=device)
            if gen_labels is not None else FN.infer_label_values(labels)
        )
        n_labels = gen_values.numel()
        bg_index = None
        if randomise_bg and (gen_values == 0).any():
            bg_index = int((gen_values == 0).nonzero(as_tuple=True)[0].item())

        # 4. GMM intensity sampling -------------------------------------------
        means, stds = FN.sample_gmm_parameters(
            n_labels, self.n_channels, batch, device,
            prior_means=self.prior_means,
            prior_stds=self.prior_stds,
            prior_distributions=self.prior_distributions,
            generation_classes=gen_classes,
            background_label_index=bg_index,
        )
        synth = FN.labels_to_image_gmm(labels, gen_values, means, stds)

        # 5. bias field -------------------------------------------------------
        if self.apply_bias_field and self.bias_field_std and self.bias_field_std > 0:
            synth = FN.bias_field(synth, self.bias_field_std, self.bias_scale)

        # 6. intensity augmentation (clip -> normalise -> gamma) --------------
        if self.apply_intensity_augmentation:
            synth = FN.intensity_augmentation(
                synth, clip=self.clip, gamma_std=self.gamma_std, normalise=self.normalise
            )

        # 7. resolution randomisation, per channel ----------------------------
        if self.apply_resolution:
            synth = self._simulate_resolution(synth)

        # 8. map generation labels -> output/segmentation labels --------------
        out_labels = labels
        if out_labels_cfg is not None and gen_labels is not None:
            out_labels = FN.convert_labels(labels, gen_labels, out_labels_cfg)

        return synth, out_labels

    # ------------------------------------------------------------------
    def _affine_active(self) -> bool:
        return any(
            b not in (False, None, 0, 0.0)
            for b in (self.scaling_bounds, self.rotation_bounds, self.shearing_bounds, self.translation_bounds)
        )

    @staticmethod
    def _random_crop(labels: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        B, _, D, H, W = labels.shape
        out = []
        sizes = (D, H, W)
        starts = []
        for dim, target in zip(sizes, output_shape):
            target = min(int(target), dim)
            start = int(torch.randint(0, dim - target + 1, (1,))) if dim > target else 0
            starts.append((start, target))
        (sd, td), (sh, th), (sw, tw) = starts
        return labels[:, :, sd:sd + td, sh:sh + th, sw:sw + tw]

    def _simulate_resolution(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        out_shape = tuple(image.shape[2:])
        atlas_res = torch.full((3,), self.atlas_res, device=device)

        channels = []
        for c in range(image.shape[1]):
            ch = image[:, c:c + 1]
            if self.randomise_res:
                res, thickness = FN.sample_resolution(
                    atlas_res, self.max_res_iso, self.max_res_aniso
                )
            else:
                res = self._fixed_res(c, device)
                thickness = self._fixed_thickness(c, device, res)
            sigma = FN.blurring_sigma_for_downsampling(atlas_res, res, thickness)
            ch = FN.gaussian_blur_3d(ch, sigma, blur_range=self.blur_range)
            ch = FN.mimic_acquisition(ch, atlas_res, res, out_shape)
            channels.append(ch)
        return torch.cat(channels, dim=1)

    def _fixed_res(self, channel: int, device: torch.device) -> torch.Tensor:
        if self.data_res is None:
            return torch.full((3,), self.atlas_res, device=device)
        arr = torch.as_tensor(self.data_res, dtype=torch.float32, device=device)
        if arr.dim() == 2:
            return arr[channel]
        return arr

    def _fixed_thickness(self, channel: int, device: torch.device, res: torch.Tensor) -> torch.Tensor:
        if self.thickness is None:
            return res
        arr = torch.as_tensor(self.thickness, dtype=torch.float32, device=device)
        if arr.dim() == 2:
            return arr[channel]
        return arr


if __name__ == "__main__":
    # Minimal self-contained smoke test on a synthetic label map (CPU-friendly).
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, D, H, W = 2, 48, 56, 52
    zz, yy, xx = torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
    )
    centre = torch.tensor([D / 2, H / 2, W / 2])
    r = ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2).sqrt()
    vol = torch.zeros(D, H, W, dtype=torch.long)
    vol[r < 18] = 1                       # "tissue A"
    vol[r < 10] = 2                       # "tissue B"
    vol[(xx > W // 2) & (r < 18)] = 3     # right-side structure
    labels = vol.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)

    gen = SynthSegGenerator(generation_labels=[0, 1, 2, 3], n_channels=1).to(device)
    image, out_labels = gen(labels.to(device))

    print("input  labels:", tuple(labels.shape), "values", torch.unique(labels).tolist())
    print("output image :", tuple(image.shape), "range",
          (round(float(image.min()), 3), round(float(image.max()), 3)))
    print("output labels:", tuple(out_labels.shape), "values", torch.unique(out_labels).tolist())
    assert image.shape[0] == B and image.shape[1] == 1
    assert out_labels.shape[2:] == image.shape[2:]
    assert not torch.isnan(image).any(), "NaNs in generated image"
    print("OK")
