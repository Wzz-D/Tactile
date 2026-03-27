from __future__ import annotations

from dataclasses import MISSING, dataclass

import torch


@dataclass
class ContactStageData:
    """Data container for per-foot contact stage filtering."""

    h_eff: torch.Tensor = MISSING
    foot_vz: torch.Tensor = MISSING
    foot_theta: torch.Tensor = MISSING

    total_force: torch.Tensor = MISSING
    total_force_filt: torch.Tensor = MISSING
    dF: torch.Tensor = MISSING
    contact_area: torch.Tensor = MISSING
    contact_area_filt: torch.Tensor = MISSING
    dA: torch.Tensor = MISSING
    cop_ap: torch.Tensor = MISSING
    rho_peak: torch.Tensor = MISSING
    rho_fore: torch.Tensor = MISSING

    contact_active: torch.Tensor = MISSING
    contact_on_event: torch.Tensor = MISSING
    contact_off_event: torch.Tensor = MISSING
    landing_window: torch.Tensor = MISSING
    tau_on: torch.Tensor = MISSING
    tau_off: torch.Tensor = MISSING
    tau_stage: torch.Tensor = MISSING
    land_trigger: torch.Tensor = MISSING
    land_maintain: torch.Tensor = MISSING
    land_active: torch.Tensor = MISSING
    stance_during_landing_scale: torch.Tensor = MISSING

    stage_eligibility: torch.Tensor = MISSING
    stage_quality: torch.Tensor = MISSING
    stage_scores: torch.Tensor = MISSING
    stage_likelihood: torch.Tensor = MISSING
    stage_prior: torch.Tensor = MISSING
    stage_belief: torch.Tensor = MISSING
    stage_confidence: torch.Tensor = MISSING
    dominant_stage_id: torch.Tensor = MISSING

    @staticmethod
    def make_zero(
        num_envs: int,
        num_feet: int,
        num_stages: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ContactStageData":
        belief = torch.full((num_envs, num_feet, num_stages), 1.0 / float(num_stages), device=device, dtype=dtype)
        return ContactStageData(
            h_eff=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            foot_vz=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            foot_theta=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            total_force=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            total_force_filt=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            dF=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            contact_area=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            contact_area_filt=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            dA=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            cop_ap=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            rho_peak=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            rho_fore=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            contact_active=torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool),
            contact_on_event=torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool),
            contact_off_event=torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool),
            landing_window=torch.zeros((num_envs, num_feet), device=device, dtype=torch.long),
            tau_on=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            tau_off=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            tau_stage=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            land_trigger=torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool),
            land_maintain=torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool),
            land_active=torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool),
            stance_during_landing_scale=torch.ones((num_envs, num_feet), device=device, dtype=dtype),
            stage_eligibility=torch.zeros((num_envs, num_feet, num_stages), device=device, dtype=dtype),
            stage_quality=torch.zeros((num_envs, num_feet, num_stages), device=device, dtype=dtype),
            stage_scores=torch.zeros((num_envs, num_feet, num_stages), device=device, dtype=dtype),
            stage_likelihood=torch.zeros((num_envs, num_feet, num_stages), device=device, dtype=dtype),
            stage_prior=torch.zeros((num_envs, num_feet, num_stages), device=device, dtype=dtype),
            stage_belief=belief,
            stage_confidence=torch.zeros((num_envs, num_feet), device=device, dtype=dtype),
            dominant_stage_id=torch.zeros((num_envs, num_feet), device=device, dtype=torch.long),
        )
