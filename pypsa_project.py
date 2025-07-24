import pypsa
import pdb as debugger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa.optimization as opt


PATH = r"C:\Users\fatih\OneDrive\Bureau\Notes cours\Athens week - european electricity grid planning\initial_csv"
SOLAR_LOAD_FACTOR_2025 = r"Post-Consultation_ERAA_2023_Climate_Data\Solar_PV\capa_factor_2025_france.csv"
# PATH = r"C:\Users\fatih\Downloads\athens-20231121T123647Z-001\athens\data\PyPSA_Europe\DATASET_RTE_N2_2050"

pdb = debugger.set_trace
excel = pd.ExcelFile(f"{PATH}\Construction_csv_pypsa.xlsm")


def generate_pypsa_data():
    filenames = ["generators", "loads", "loads-p_set", "network",
                 "storage_units", "buses", "carriers", "links", "snapshots", "store"]
    for filename in filenames:
        filepath = f"{PATH}\{filename}.csv"
        df = pd.read_excel(excel, sheet_name=filename)
        df = df.rename(columns={"Unnamed: 0": np.nan})
        if filename != "links":
            df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if filename == 'network':
            df = pd.concat([pd.DataFrame(columns=['name']), df], axis=0)
        df.to_csv(filepath, index=False)


def generate_timeseries(input: str, resample_fn: str = 'mean', save: bool = False, target: str = None) -> pd.DataFrame:
    path = r"C:\Users\fatih\Downloads\athens-20231121T123647Z-001\athens\data\ERAA\eraa_preprocess_2023-11-19_105243\{}".format(
        input)
    df = pd.read_csv(path, sep=';')
    df.index = pd.DatetimeIndex(df.date)
    resampled_obj = df[['value']].resample("1D")
    if resample_fn == 'min':
        df = resampled_obj.min()
    elif resample_fn == 'max':
        df = resampled_obj.max()
    else:
        df = resampled_obj.mean()
    if save:
        df.to_csv(target, index=False)
    return df


generate_pypsa_data()


def resample_france_data():
    path = r"C:\Users\fatih\Downloads\athens-20231121T123647Z-001\athens\data\ERAA\eraa_preprocess_2023-11-19_105243\Post-Consultation_ERAA_2023_Demand_data\Demand_TimeSeries\demand_2025_france.csv"
    target = r"C:\Users\fatih\OneDrive\Bureau\Notes cours\Athens week - european electricity grid planning\rte_2025_demand.csv"
    df = pd.read_csv(path, sep=';')
    df.index = pd.DatetimeIndex(df.date)
    df = df[['value']].resample("1D").max()
    df.to_csv(target, index=False)


# resample_france_data()
network = pypsa.Network()
network.import_from_csv_folder(csv_folder_name=PATH)
load_factor = generate_timeseries(SOLAR_LOAD_FACTOR_2025)

pdb()

# network.plot(
#     title="France network (1 node)",
#     color_geomap=True,
#     jitter=0.3,
# )
# plt.show()


# Optimisation step:
network.optimize(solver_name='gurobi')
model = opt.create_model(network)
model.to_file(f'{PATH}\model.lp')
pdb()
plt.clf()
network.generators.p_nom_opt.div(1e3).plot.bar(ylabel="GW", figsize=(8, 3))
plt.tight_layout()
plt.show()

plt.clf()
network.generators_t.p.div(1e3).plot.area(subplots=True, ylabel="GW")
plt.tight_layout()
plt.show()
