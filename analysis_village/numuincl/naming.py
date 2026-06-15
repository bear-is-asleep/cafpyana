PRELIM_LABEL = 'SBND Preliminary Simulation'
PRELIM_DATA_LABEL = 'SBND Preliminary'
INTERNAL_LABEL = 'SBND Internal Simulation'
INTERNAL_DATA_LABEL = 'SBND Internal'

#PANDORA_QUALIFIER = r'Pandora TPC reconstruction ($\chi^2$ cut, by TPC)'
#PANDORA_QUALIFIER = r'Algorithm-based TPC reconstruction, local $\chi^2$ optical cut'
PANDORA_QUALIFIER = r'Flash Match ($\chi^2$ cut, by TPC) + hit-based TPC reconstruction'
PANDORA_QUALIFIER = f'Pandora '
PANDORA_QUALIFIER_PRELIM_LABEL = PANDORA_QUALIFIER + PRELIM_LABEL
PANDORA_QUALIFIER_INTERNAL_LABEL = PANDORA_QUALIFIER + INTERNAL_LABEL
PANDORA_QUALIFIER_INTERNAL_LABEL = PANDORA_QUALIFIER + INTERNAL_LABEL
PANDORA_QUALIFIER_PRELIM_DATA_LABEL = PANDORA_QUALIFIER + PRELIM_DATA_LABEL
PANDORA_QUALIFIER_INTERNAL_DATA_LABEL = PANDORA_QUALIFIER + INTERNAL_DATA_LABEL

#SPINE_QUALIFIER = r'SPINE TPC reconstruction (best match, whole detector)'
SPINE_QUALIFIER = 'Flash Match (best match, whole detector) + SPINE (ML) TPC reconstruction'
SPINE_QUALIFIER = f'SPINE '
SPINE_QUALIFIER_PRELIM_LABEL = SPINE_QUALIFIER + PRELIM_LABEL
SPINE_QUALIFIER_INTERNAL_LABEL = SPINE_QUALIFIER + INTERNAL_LABEL
SPINE_QUALIFIER_INTERNAL_LABEL = SPINE_QUALIFIER + INTERNAL_LABEL
# Cut lists (canonical order for detsys / contained analyses)
PAND_CUTS_BASE = ['flashpe', 'flashmatch', 'cosmic', 'fv', 'muon']
PAND_CUT_LABELS_BASE = [
    'Flash PE > 2000',
    'Flash Match\n(best BCFM)',
    'Flash Score\n(score cut)',
    'Fiducial',
    'Has Muon',
]

PAND_CUTS = PAND_CUTS_BASE + ['lowz']
PAND_CUT_LABELS = PAND_CUT_LABELS_BASE + ['Low Z']

PAND_CUTS_CONT = PAND_CUTS_BASE + ['cont_full', 'cont', 'all_cont']
PAND_CUT_LABELS_CONT = PAND_CUT_LABELS_BASE + [
    'Muon Contained\n(full detector)',
    'Muon Contained\n(by TPC)',
    'Slice Contained\n(by TPC)',
]

GENIE_LABEL = 'GENIE v3.4.0 AR23_20i_00_000'

#Key structure
MCNU_KEY = 'mcnu*'
PAND_KEY = 'evt_pand*'
PAND_SELECTED_KEY = 'evt_pand_selected*'
PAND_SIGNAL_KEY = 'evt_pand_signal*'
HDR_KEY = 'hdr*'
POT_KEY = 'histpotdf*'
GENEVT_KEY = 'histgenevtdf*'
PFP_KEY = 'trk*'

#Data location
DATA_DIR = '/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora'