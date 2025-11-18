import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
from autorank import autorank, create_report, plot_stats, latex_table
if __name__ == '__main__':
    print("NMI real world statistical test:")
    nmi_realWorld={"karate":[0.60,0.94,0.69,0.63,1.0,0.58,0.60,0.54,1.0,0.69,0.59,0.69,0.70,0.70],
                   "polbooks":[0.57,0.76,0.53,0.40,0.56,0.48,0.53,0.43,0.55,0.52,0.51,0.53,0.49,0.57],
                   "polblogs": [0.45, 0.64,0.33,0.30,0.48,0.45,0.36,0.28,0.68,0.63,0.63,0.65,0.48,0.69],
                   "football": [0.82,0.89,0.70,0.91,0.89,0.89,0.88,0.69,0.89,0.87,0.88,0.70,0.92,0.92],
                   "dolphins": [0.73,0.90,0.73,0.58,0.73,0.46,0.78,0.69,0.73,0.77,0.48,0.61,0.50,0.69]}

    dataframe_nmi=pd.DataFrame.from_dict(nmi_realWorld,orient='index',columns=["proposed","CACD","CCGA","GA_net","WATSET","FluidCom","EdMot","WMW","LocalGame","Leiden","Louvain","Fastgreedy","Infomap","LP"])
    general_dataframe_nmi = dataframe_nmi

    # l_data = pd.melt(general_dataframe_nmi, var_name='criteria', value_name='score')
    # p_values_general_nmi = sp.posthoc_dunn(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # p_values_general_nmi = sp.posthoc_wilcoxon(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # res = friedmanchisquare(general_dataframe_nmi["proposed"],general_dataframe_nmi["CACD"],general_dataframe_nmi["CCGA"],
    #                         general_dataframe_nmi["GA_net"],general_dataframe_nmi["WATSET"],
    #                         general_dataframe_nmi["FluidCom"], general_dataframe_nmi["EdMot"],
    #                         general_dataframe_nmi["WMW"], general_dataframe_nmi["LocalGame"],
    #                         general_dataframe_nmi["Leiden"],general_dataframe_nmi["Louvain"]
    #                         ,general_dataframe_nmi["Fastgreedy"],general_dataframe_nmi["Infomap"],
    #                         general_dataframe_nmi["LP"])
    # print("friedman test nmi: ")
    # print(res)
    # print("algorithms p_values")
    # print(p_values_general_nmi)

    res = autorank(general_dataframe_nmi, alpha=0.05, verbose=True)
    print(res)
    create_report(res)
    plot_stats(res)
    plt.show()
    latex_table(res)

    print("===============================================")
    print("Q real world statistical test:")
    q_realWorld = {"karate": [0.42,0.42,0.42,0.40,0.37,0.35,0.42,0.40,0.37,0.42,0.42,0.38,0.40,0.40],
                   "polbooks": [0.52,0.53,0.52,0.43,0.49,0.44,0.53,0.47,0.51,0.52,0.52,0.50,0.52,0.50],
                   "polblogs": [0.43,0.34,0.37,0.35,0.43,0.43,0.29,0.40,0.40,0.43,0.43,0.43,0.42,0.43],
                   "football": [0.60,0.58,0.55,0.60,0.58,0.58,0.60,0.39,0.58,0.60,0.60,0.55,0.60,0.60],
                   "dolphins": [0.52,0.52,0.52,0.42,0.48,0.38,0.52,0.49,0.51,0.53,0.52,0.50,0.52,0.50],
                   "Power": [0.93,0.77,0.90,0.64,0.79,0.89,0.94,0.61,0.90,0.94,0.93,0.93,0.82,0.81],
                   "Jazz": [0.44,0.43,0.42,0.37,0.28,0.42,0.44,0.32,0.28,0.44,0.44,0.44,0.28,0.28],
                   "PGP": [0.86,0.79,0.84,0.60,0.80,0.82,0.87,0.67,0.79,0.88,0.88,0.85,0.80,0.81],
                   "caHep": [0.75,0.66,0.53,0.53,0.69,0.68,0.77,0.57,0.63,0.77,0.82,0.78,0.73,0.74]}

    dataframe_q = pd.DataFrame.from_dict(q_realWorld, orient='index',
                                       columns=["proposed", "CACD", "CCGA", "GA_net", "WATSET", "FluidCom", "EdMot",
                                                "WMW", "LocalGame", "Leiden", "Louvain", "Fastgreedy", "Infomap", "LP"])
    general_dataframe_q = dataframe_q

    # l_data = pd.melt(general_dataframe_q, var_name='criteria', value_name='score')
    # p_values_general_q = sp.posthoc_dunn(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # p_values_general_q =sp.posthoc_wilcoxon(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # res = friedmanchisquare(general_dataframe_q["proposed"],general_dataframe_q["CACD"],general_dataframe_q["CCGA"],
    #                         general_dataframe_q["GA_net"],general_dataframe_q["WATSET"],
    #                         general_dataframe_q["FluidCom"], general_dataframe_q["EdMot"],
    #                         general_dataframe_q["WMW"], general_dataframe_q["LocalGame"],
    #                         general_dataframe_q["Leiden"],general_dataframe_q["Louvain"]
    #                         ,general_dataframe_q["Fastgreedy"],general_dataframe_q["Infomap"],
    #                         general_dataframe_q["LP"])
    # print("friedman test q: ")
    # print(res)
    # print("algorithms p_values")
    # print(p_values_general_q)
    res = autorank(general_dataframe_q, alpha=0.05, verbose=False)
    print(res)
    create_report(res)
    plot_stats(res)
    plt.show()
    latex_table(res)
    print("===============================================")
    print("NMI synthetic statistical test:")

    nmi_synthetic = {
    "LFR_1": [0.54,0.80,0.56,0.43,0.65,0.80,0.61,0.43,1.0,0.61,0.39,0.66,0.54,0.80],
    "LFR_2": [1.0,1.0,0.33,0.42,1.0,1.0,1.0,0.42,1.0,1.0,1.0,1.0,1.0,1.0],
    "LFR_3": [1.0,1.0,0.75,0.85,1.0,0.82,1.0,0.96,1.0,1.0,1.0,0.99,1.0,1.0],
    "LFR_4": [1.0, 0.87,0.37,0.30,1.0,0.86,1.0,1.0,1.0,1.0,1.0,0.99,1.0,1.0],
    "LFR_5": [1.0, 0.57,0.2,0.1,1.0,0.44,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
    "LFR_6": [1.0, 0.48, 0.1, 0.1, 1.0, 0.59, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    dataframe_nmi_synthetic = pd.DataFrame.from_dict(nmi_synthetic, orient='index',
                                       columns=["proposed", "CACD", "CCGA", "GA_net", "WATSET", "FluidCom", "EdMot",
                                                "WMW", "LocalGame", "Leiden", "Louvain", "Fastgreedy", "Infomap", "LP"])
    general_dataframe_nmi_synthetic = dataframe_nmi_synthetic

    # l_data = pd.melt(general_dataframe_nmi_synthetic, var_name='criteria', value_name='score')
    # p_values_general_nmi_synthetic = sp.posthoc_dunn(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # p_values_general_nmi_synthetic =sp.posthoc_wilcoxon(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # res = friedmanchisquare(general_dataframe_nmi_synthetic["proposed"],general_dataframe_nmi_synthetic["CACD"]
    #                         ,general_dataframe_nmi_synthetic["CCGA"], general_dataframe_nmi_synthetic["GA_net"]
    #                         ,general_dataframe_nmi_synthetic["WATSET"],general_dataframe_nmi_synthetic["FluidCom"]
    #                         , general_dataframe_nmi_synthetic["EdMot"],general_dataframe_nmi_synthetic["WMW"]
    #                         , general_dataframe_nmi_synthetic["LocalGame"],general_dataframe_nmi_synthetic["Leiden"]
    #                         ,general_dataframe_nmi_synthetic["Louvain"],general_dataframe_nmi_synthetic["Fastgreedy"]
    #                         ,general_dataframe_nmi_synthetic["Infomap"],general_dataframe_nmi_synthetic["LP"])
    # print("friedman test general_nmi synthetic: ")
    # print(res)
    # print("algorithms p_values synthetic")
    # print(p_values_general_nmi_synthetic)

    res = autorank(general_dataframe_nmi_synthetic, alpha=0.05, verbose=False)
    print(res)
    create_report(res)
    plot_stats(res)
    plt.show()
    latex_table(res)

    print("===============================================")
    print("Q synthetic statistical test:")

    q_synthetic = {
    "LFR_1": [0.61,0.54,0.59,0.52,0.59,0.54,0.62,0.52,0.49,0.50,0.47,0.62,0.60,0.56],
    "LFR_2": [0.43,0.43,0.24,0.24,0.43,0.43,0.43,0.24,0.43,0.43,0.43,0.43,0.43,0.43],
    "LFR_3": [0.82,0.82,0.64,0.61,0.82,0.74,0.82,0.74,0.82,0.82,0.82,0.81,0.82,0.82],
    "LFR_4": [0.70,0.58,0.10,0.20,0.70,0.58,0.70,0.70,0.70,0.70,0.70,0.69,0.70,0.70],
    "LFR_5": [0.53,0.11,0.10,0.10,0.53,0.10,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53],
    "LFR_6": [0.66,0.11,0.10,0.10,0.66,0.18,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66]
    }

    dataframe_q_synthetic = pd.DataFrame.from_dict(q_synthetic, orient='index',
                                       columns=["proposed", "CACD", "CCGA", "GA_net", "WATSET", "FluidCom", "EdMot",
                                                "WMW", "LocalGame", "Leiden", "Louvain", "Fastgreedy", "Infomap", "LP"])
    general_dataframe_q_synthetic = dataframe_q_synthetic
    # l_data = pd.melt(general_dataframe_q_synthetic, var_name='criteria', value_name='score')
    # p_values_general_q_synthetic = sp.posthoc_dunn(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # p_values_general_q_synthetic = sp.posthoc_wilcoxon(l_data, val_col='score', group_col='criteria', p_adjust='bonferroni')
    # res = friedmanchisquare(general_dataframe_q_synthetic["proposed"],general_dataframe_q_synthetic["CACD"]
    #                         ,general_dataframe_q_synthetic["CCGA"], general_dataframe_q_synthetic["GA_net"]
    #                         ,general_dataframe_q_synthetic["WATSET"],general_dataframe_q_synthetic["FluidCom"]
    #                         , general_dataframe_q_synthetic["EdMot"],general_dataframe_q_synthetic["WMW"]
    #                         , general_dataframe_q_synthetic["LocalGame"],general_dataframe_q_synthetic["Leiden"]
    #                         ,general_dataframe_q_synthetic["Louvain"],general_dataframe_q_synthetic["Fastgreedy"]
    #                         ,general_dataframe_q_synthetic["Infomap"],general_dataframe_q_synthetic["LP"])
    # print("friedman test general_q synthetic: ")
    # print(res)
    # print("general algorithms p_values synthetic")
    # print(p_values_general_q_synthetic)
    res = autorank(general_dataframe_q_synthetic, alpha=0.05, verbose=False)
    print(res)
    create_report(res)
    plot_stats(res)
    plt.show()
    latex_table(res)