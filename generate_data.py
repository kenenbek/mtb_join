import os
import cobra
from cobra.io import load_json_model
import cobrascape.species as cs


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import scipy.optimize as opt


ENSEMBLE_DIR = "./mnc_ensemble_0/"
DATA_DIR = "./input_data/"
MODEL_FILE = DATA_DIR + 'MODEL_FILE.json'
X_ALLELES_FILE = DATA_DIR + 'X_ALLELES_FILE.csv'            # "iEK1011_drugTesting_media.json"
Y_PHENOTYPES_FILE = DATA_DIR + 'Y_PHENOTYPES_FILE.csv'      #
GENE_LIST_FILE = DATA_DIR + "GENE_LIST_FILE.csv" # If not found, will use ALL genes in cobra model
MODEL_SAMPLES_FILENAME = "base_flux_samples.csv" # If not found in ENSEMBLE_DIR, script will perform flux sampling.

OPEN_EXCHANGE_FVA = True
FVA_CONSTRAINTS = True

FVA_frac_opt = 0.0
FVA_pfba_fract = 1.1

GENERATE_SAMPLES = True

X_species = pd.read_csv(X_ALLELES_FILE, index_col = 0)
Y_phenotypes = pd.read_csv(Y_PHENOTYPES_FILE, index_col = 0)

# X_df = X_species.copy()
# Y_df = Y_phenotypes.reindex(X_df.index.tolist()).copy()
X_train, X_test, y_train, y_test = train_test_split(X_species,
                                                    Y_phenotypes.reindex(X_species.index.tolist()),
                                                    test_size=0.3,
                                                    random_state=42)
if not os.path.exists(ENSEMBLE_DIR+"/X_train.csv"):
    X_train.to_csv(ENSEMBLE_DIR+"/X_train.csv")
    X_test.to_csv(ENSEMBLE_DIR+"/X_test.csv")
    y_train.to_csv(ENSEMBLE_DIR+"/y_train.csv")
    y_test.to_csv(ENSEMBLE_DIR+"/y_test.csv")

X_species_final = X_train
Y_pheno_final = y_train
print("input: (G)enetic variant matrix= (strains: %d, alleles: %d)" % (X_species_final.shape[0], X_species_final.shape[1]))
print("input: Class distribution for each phenotype")
for pheno in Y_pheno_final.columns:
    print("\t",pheno, "train:", (y_train[pheno].value_counts().to_dict()),
          "test:", (y_test[pheno].value_counts().to_dict()))


COBRA_MODEL = load_json_model(MODEL_FILE)
print("input: (S)toichimetric genome-scale model= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes),
    len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))

### The desired media condition should already be initialized.
print("COBRA_MODEL.medium: ", COBRA_MODEL.medium)

sol = COBRA_MODEL.optimize()
print("\t... before cleaning (objective_value: %f)" % (sol.objective_value))
### Clean base model and apply FVA constriants
COBRA_MODEL = cs.clean_base_model(COBRA_MODEL, open_exchange=OPEN_EXCHANGE_FVA, verbose=False)
sol = COBRA_MODEL.optimize()
print("\t... after cleaning (objective_value: %f)" % (sol.objective_value))

if FVA_CONSTRAINTS==True:
    COBRA_MODEL, fva_df = cs.init_fva_constraints(COBRA_MODEL,opt_frac=FVA_frac_opt, pfba_fact=FVA_pfba_fract, verbose=False)
    sol = COBRA_MODEL.optimize()
    print("\t... after fva constraints (objective_value: %f)" % (sol.objective_value))
    print("\t... filtered GEM= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes),
        len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))


# ENSEMBLE_DIR = "ens_strains"+str(len(SPECIES_MODEL.strains))+"_alleles"+str(len(players))+"_actions"+str(action_num)
POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
print("output dir: %s" % (ENSEMBLE_DIR))
if not os.path.exists(POPFVA_SAMPLES_DIR):
    print('\t... creating sampling directory:'+POPFVA_SAMPLES_DIR)
    os.makedirs(POPFVA_SAMPLES_DIR)


pheno_list = Y_pheno_final.columns
ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"

### ------------------------------------------------------------
### Sample allele-constraint maps and popFVA landscapes for MNCs
### ------------------------------------------------------------
if GENERATE_SAMPLES == True:

    MODEL_SAMPLES_FILE = ENSEMBLE_DIR + "/" + MODEL_SAMPLES_FILENAME
    if not os.path.exists(MODEL_SAMPLES_FILE):
        from cobra import sampling

        print("\t... generating flux samples for base cobra model...(may take >10 minutes). Only performed once!")
        rxn_flux_samples_ARCH = sampling.sample(COBRA_MODEL, 10000, method='achr',
                                                thinning=100, processes=6, seed=None)
        print("\t... saving flux samples for base cobra model: ", MODEL_SAMPLES_FILE)
        rxn_flux_samples_ARCH.to_csv(MODEL_SAMPLES_FILE)

    ENSEMBLE_BASEMODEL_FILE = ENSEMBLE_DIR + "/base_cobra_model.json"
    if not os.path.exists(ENSEMBLE_BASEMODEL_FILE):
        from cobra.io import save_json_model

        print("\t... saving base cobra model: ", ENSEMBLE_BASEMODEL_FILE)
        save_json_model(COBRA_MODEL, ENSEMBLE_BASEMODEL_FILE)

    base_flux_samples = pd.read_csv(MODEL_SAMPLES_FILE, index_col=0)

min_values = base_flux_samples.min()
max_values = base_flux_samples.max()

reaction_stats = pd.DataFrame({
    'Reaction': base_flux_samples.columns,
    'Min': min_values,
    'Max': max_values
})

# Optionally, save this DataFrame to a CSV file
reaction_stats.to_csv('reaction_min_max_values.csv', index=False)

c_dict = {reaction.id: reaction.objective_coefficient for reaction in COBRA_MODEL.reactions}
#c_dict = dict(sorted(c_dict.items()))
c_vector = np.array(list(c_dict.values()))

# Extract the stoichiometric matrix
S_matrix = cobra.util.create_stoichiometric_matrix(COBRA_MODEL)

N = 5
reaction_bounds_all = [[] for _ in range(N)]
reaction_bounds_all_dict = [[] for _ in range(N)]

for index, row in reaction_stats.iterrows():
    min_val = row["Min"]
    max_val = row["Max"]
    reaction = row["Reaction"]

    for i in range(N):
        start = np.random.uniform(min_val, max_val)
        end = np.random.uniform(start, max_val)

        reaction_bounds_all_dict[i].append({reaction: [start, end]})
        reaction_bounds_all[i].append([start, end])

reaction_bounds_all = [np.array(reaction_bounds) for reaction_bounds in reaction_bounds_all]

results = []

for reaction_bounds in reaction_bounds_all:
    result = opt.linprog(c_vector,
                         A_ub=None,
                         b_ub=None,
                         A_eq=S_matrix,
                         b_eq=np.zeros(S_matrix.shape[0]),
                         bounds=reaction_bounds)

    results.append(result)