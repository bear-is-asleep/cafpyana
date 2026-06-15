from . import getsyst
import pandas as pd
import numpy as np

#AR23+ systematic variations
r23p_genie_systematics = [
    'ZExpPCAWeighter_SBNNuSyst_multisigma_MvA_ZExp_b1',
    'ZExpPCAWeighter_SBNNuSyst_multisigma_MvA_ZExp_b2',
    'ZExpPCAWeighter_SBNNuSyst_multisigma_MvA_ZExp_b3',
    'ZExpPCAWeighter_SBNNuSyst_multisigma_MvA_ZExp_b4',
    'CCQETemplateReweight_SBNNuSyst_multisigma_SF_q0bin1',
    'CCQETemplateReweight_SBNNuSyst_multisigma_SF_q0bin2',
    'CCQETemplateReweight_SBNNuSyst_multisigma_SF_q0bin3',
    'CCQETemplateReweight_SBNNuSyst_multisigma_SF_q0bin4',
    'CCQETemplateReweight_SBNNuSyst_multisigma_SF_q0bin5',
    'CCQETemplateReweight_SBNNuSyst_multisigma_CRPA_q0bin1',
    'CCQETemplateReweight_SBNNuSyst_multisigma_CRPA_q0bin2',
    'CCQETemplateReweight_SBNNuSyst_multisigma_CRPA_q0bin3',
    'CCQETemplateReweight_SBNNuSyst_multisigma_CRPA_q0bin4',
    'QEInterference_SBNNuSyst_multisigma_INT_QEIntf_dial_0',
    'QEInterference_SBNNuSyst_multisigma_INT_QEIntf_dial_1',
    'QEInterference_SBNNuSyst_multisigma_INT_QEIntf_dial_2',
    'QEInterference_SBNNuSyst_multisigma_INT_QEIntf_dial_3',
    'QEInterference_SBNNuSyst_multisigma_INT_QEIntf_dial_4',
    'QEInterference_SBNNuSyst_multisigma_INT_QEIntf_dial_5',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_VecFFCCQEshape',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_CoulombCCQE',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_NormCCMEC',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_NormNCMEC',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_DecayAngMEC',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_MFP_pi',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_FrCEx_pi',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_FrInel_pi',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_FrAbs_pi',
    'GENIEReWeight_SBNNuSyst_multisigma_EDepFSI_FrPiProd_pi',
    'MECq0q3InterpWeighting_SuSAv2ToValenica_q0binned_MECResponse_q0bin0',
    'MECq0q3InterpWeighting_SuSAv2ToValenica_q0binned_MECResponse_q0bin1',
    'MECq0q3InterpWeighting_SuSAv2ToValenica_q0binned_MECResponse_q0bin2',
    'MECq0q3InterpWeighting_SuSAv2ToValenica_q0binned_MECResponse_q0bin3',
    'MECq0q3InterpWeighting_SuSAv2ToMartini_q0binned_MECResponse_q0bin0',
    'MECq0q3InterpWeighting_SuSAv2ToMartini_q0binned_MECResponse_q0bin1',
    'MECq0q3InterpWeighting_SuSAv2ToMartini_q0binned_MECResponse_q0bin2',
    'MECq0q3InterpWeighting_SuSAv2ToMartini_q0binned_MECResponse_q0bin3'
]


# grouped syst knobs
qe_genie_systematics = [
    'GENIEReWeight_SBN_v1_multisim_RPA_CCQE',
    'GENIEReWeight_SBN_v1_multisim_CoulombCCQE',
]

mec_genie_systematics = [
    'GENIEReWeight_SBN_v1_multisim_NormCCMEC',
    'GENIEReWeight_SBN_v1_multisim_NormNCMEC',
    "GENIEReWeight_SBN_v1_multisigma_DecayAngMEC",
]

res_genie_systematics = [
    'GENIEReWeight_SBN_v1_multisim_RDecBR1gamma',
    'GENIEReWeight_SBN_v1_multisim_RDecBR1eta',
    "GENIEReWeight_SBN_v1_multisigma_Theta_Delta2Npi",
    "GENIEReWeight_SBN_v1_multisigma_ThetaDelta2NRad",

    "GENIEReWeight_SBN_v1_multisigma_MaCCRES",
    "GENIEReWeight_SBN_v1_multisigma_MaNCRES",
    "GENIEReWeight_SBN_v1_multisigma_MvCCRES",
    "GENIEReWeight_SBN_v1_multisigma_MvNCRES",
]

nonres_genie_systematics = [
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpNC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnNC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpNC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnNC2pi',
]

dis_genie_systematics = [
    'GENIEReWeight_SBN_v1_multisigma_AhtBY',
    'GENIEReWeight_SBN_v1_multisigma_BhtBY',
    'GENIEReWeight_SBN_v1_multisigma_CV1uBY',
    'GENIEReWeight_SBN_v1_multisigma_CV2uBY',
]

other_genie_systematics = [
    'GENIEReWeight_SBN_v1_multisigma_MFP_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrCEx_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrInel_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrAbs_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrPiProd_pi',
    'GENIEReWeight_SBN_v1_multisigma_MFP_N',
    'GENIEReWeight_SBN_v1_multisigma_FrCEx_N',
    'GENIEReWeight_SBN_v1_multisigma_FrInel_N',
    'GENIEReWeight_SBN_v1_multisigma_FrAbs_N',
    'GENIEReWeight_SBN_v1_multisigma_FrPiProd_N',
    "GENIEReWeight_SBN_v1_multisigma_NormCCCOH", # Handled by re-tuning
    "GENIEReWeight_SBN_v1_multisigma_NormNCCOH",
    'GENIEReWeight_SBN_v1_multisigma_MaNCEL',
    'GENIEReWeight_SBN_v1_multisigma_EtaNCEL',
]

_full_genie_systematics = (
    r23p_genie_systematics + qe_genie_systematics + mec_genie_systematics
    + res_genie_systematics + nonres_genie_systematics + dis_genie_systematics
    + other_genie_systematics
)

# Legacy alias (slim and full both use AR23+ union, incl. MvA_ZExp)
gen1_systematics = _full_genie_systematics

GENIE_KNOB_GROUPS = {
    "Ar23p": r23p_genie_systematics,
    "CCQE": qe_genie_systematics,
    "MEC": mec_genie_systematics,
    "RES": res_genie_systematics,
    "nonRES": nonres_genie_systematics,
    "DIS": dis_genie_systematics,
    "Other": other_genie_systematics,
    "gen1": gen1_systematics,
}


_MVA_ZEXP_SUBSTR = "MvA_ZExp"
_SLIM_MULTISIGMA_BUNDLE = "GENIE_multisigma"


def _is_mva_zexp_knob(name: str) -> bool:
    return _MVA_ZEXP_SUBSTR in str(name)


def _bundle_slim_multisigma(df: pd.DataFrame) -> pd.DataFrame:
    """Product non-MvA ±σ / morph leaves into ``GENIE_multisigma``; keep MvA per-knob."""
    if df.empty:
        return df

    multisim_cols = []
    mva_cols = []
    bundle_leaves: dict[str, np.ndarray] = {}

    for c in df.columns:
        if not isinstance(c, tuple) or len(c) < 2:
            continue
        top, leaf = str(c[0]), str(c[1])
        if top == "GENIE" and leaf.startswith("univ_"):
            multisim_cols.append(c)
            continue
        if _is_mva_zexp_knob(top):
            mva_cols.append(c)
            continue
        if leaf.startswith("univ_"):
            continue
        arr = df[c].to_numpy(dtype=float, copy=False)
        if leaf not in bundle_leaves:
            bundle_leaves[leaf] = np.ones(len(df), dtype=float)
        bundle_leaves[leaf] *= arr

    parts = []
    if multisim_cols:
        parts.append(df[multisim_cols])
    if bundle_leaves:
        parts.append(
            pd.DataFrame(
                {
                    (_SLIM_MULTISIGMA_BUNDLE, leaf): vals
                    for leaf, vals in bundle_leaves.items()
                },
                index=df.index,
            )
        )
    if mva_cols:
        parts.append(df[mva_cols])
    if not parts:
        return df
    return pd.concat(parts, axis=1)


def per_knob_names_in_mc_weights(mc_evt_df) -> tuple:
    """Knob names with ±σ / morph leaves under ``mc`` (excludes bundled ``GENIE``)."""
    cols = mc_evt_df.columns
    if not isinstance(cols, pd.MultiIndex) or cols.nlevels < 3:
        return ()
    try:
        mc = mc_evt_df.mc
    except (AttributeError, KeyError):
        return ()
    knobs = []
    for knob in mc.columns.get_level_values(0).unique():
        if knob in (None, "", "GENIE", _SLIM_MULTISIGMA_BUNDLE):
            continue
        leaves = {
            str(c[0]) if isinstance(c, tuple) else str(c) for c in mc[knob].columns
        }
        if leaves & {"ps1", "ms1", "morph"} or any(x.startswith("univ_") for x in leaves):
            knobs.append(knob)
    return tuple(sorted(knobs))


def geniesyst(f, nuind, multisim_nuniv=100, slim=False, systematics=None):
    if systematics is None:
        systematics = _full_genie_systematics

    geniewgtdf = getsyst.getsyst(f, systematics, nuind, multisim_nuniv=multisim_nuniv, slim=slim, slimname="GENIE")
    geniewgtdf = geniewgtdf.clip(lower=0, upper=10)
    if slim:
        geniewgtdf = _bundle_slim_multisigma(geniewgtdf)
    return geniewgtdf