import os
import pathlib
import statistics
from collections import OrderedDict

import pathlib as pl
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
from f_evolution_tools import *


table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}

dpe_colors = ['#009900', '#33cc33', '#B3FF00', '#e6e600', '#FFB300', '#FF4D00', '#FF0000', "#000000"]

data_set_from_excel = pd.read_excel("data/Hypotheses_ResTer_Heating_Scenario1.xlsx", None);
#data_set_from_excel['year_res_type']:   building_type  year  retrofit_improvement
#data_set_from_excel["date_step"]=5

def Compute_param_and_launch_simulation(data_set_from_excel):
    dim_names = ["Energy_source", "building_type", "Vecteur", "year"];
    Index_names = ["Energy_source", "building_type"];
    Energy_system_name = "Energy_source"
    sim_param = extract_sim_param(data_set_from_excel, Index_names=Index_names, dim_names=dim_names,
                                  Energy_system_name=Energy_system_name)
    # Creating the initial building description
    sim_param["init_sim_stock"] = create_initial_parc(sim_param).sort_index()
    sim_param["volume_variable_name"] = "surface"
    sim_param["init_sim_stock"]["surface"] = sim_param["init_sim_stock"]["surface"] * sim_param["init_sim_stock"][
        "IPONDL"]
    sim_param = interpolate_sim_param(sim_param)
    sim_param["retrofit_change_surface"] = sim_param["retrofit_change_total_proportion_surface"].diff().fillna(0)
    Para_2_fill = {param: sim_param["base_index_year"] for param in
                   ["retrofit_improvement", "retrofit_change_surface", "retrofit_Transition", "new_yearly_surface",
                    "new_energy"]}
    sim_param = complete_parameters(sim_param, Para_2_fill=Para_2_fill)

    final_share = (1 - sim_param["old_taux_disp"].sum()) if "old_taux_disp" in sim_param else 1

    sim_param["retrofit_change_surface"] = sim_param["retrofit_change_surface"] * sim_param["init_sim_stock"][
        "surface"] * final_share

    # When data is not given for every string index (typically vecteurs), we complete
    sim_param = complete_missing_indexes(data_set_from_excel, sim_param, Index_names, dim_names)

    ## We define some functions which will calculate at each time step the energy need, consumption, emissions...
    sim_param = set_model_functions(sim_param)

    # We lanch the simulation
    sim_stock = launch_simulation(sim_param)
    return sim_stock,sim_param,Energy_system_name







app = Dash(__name__,external_stylesheets=[dbc.themes.LUMEN])
app.title = "Pharmacokinetics Calculator"
server = app.server

APP_PATH = str(pl.Path(__file__).parent.resolve())

columns_type = {"building_type" : "text","year" : "numeric","retrofit_improvement" : "numeric",
                'Energy_source': "text", 'building_type': "text", 'Biomasse': "numeric", 'Chaudière fioul': "numeric",
               'Chaudière gaz': "numeric", 'Chauffage urbain': "numeric", 'Pompes à chaleur air-air': "numeric",
               'Pompes à chaleur hybride': "numeric", 'Chauffage électrique': "numeric",
               'Pompes à chaleur air-eau': "numeric"
}

columns_names={}; columns_table={}
tables = ['retrofit_Transition','year_res_type']
for table in tables:
    columns_names[table] = data_set_from_excel[table].columns.to_list()
    columns_table[table] = [{"name" : i, "id" : i,"type" : columns_type[i]} for i in columns_names[table]]

anchor_id = {}
table_text ={}

anchor_id['retrofit_Transition'] = "2.1hypothese"
table_text['retrofit_Transition'] = '''
    ## 2.1  Modélisation de la transition des modes de chauffage 
'''

anchor_id['year_res_type'] = "2.2hypothese"
table_text['year_res_type'] = '''
    ## 2.2  Modélisation de la performance des rénovations
'''

Graphiques_Etat_actuel = ['output_actuel_conso','output_actuel_emissions']
Graphiques_simulation = ['conso','conso_finale','emissions']


anchor_id['output_actuel_conso'] = "3.1.1actuel"
table_text['output_actuel_conso']='''
    ## 3.1.1  Etat actuel du parc - consommation
'''
anchor_id['output_actuel_emissions'] = "3.1.2actuel"
table_text['output_actuel_emissions']='''
    ## 3.1.2  Etat actuel du parc - emissions
'''

anchor_id['conso'] = "3.2.1prospective"
table_text['conso']='''
    ## 3.2.1  Résultats de la simulation prospective - consommation par type de chauffage
'''

anchor_id['conso_finale'] = "3.2.2prospective"
table_text['conso_finale']='''
    ## 3.2.2  Résultats de la simulation prospective - consommation par vecteur 
'''

anchor_id['emissions'] = "3.2.3prospective"
table_text['emissions']='''
    ## 3.2.3  Résultats de la simulation prospective - emissions
'''

table_of_content='''
    ### Table of Contents
    
    * [1. Introduction](#intro)
    * [2. Hypothèses du parc bâti](#initialstate)
        * [2.1 Modélisation de la transition des modes de chauffage](#2.1hypothese)
        * [2.2 Modélisation de la performance des rénovations](#2.2hypothese)    
    * [3. Résultat de la simulation](#simulation)
        * [3.1 Etat actuel du parc](#3.1actuel)
            * [3.1.1 consommation](#3.1.1actuel)
            * [3.1.2 consommation](#3.1.2actuel)
        * [3.2 Résultats de la simulation prospective](#prospective)
            * [3.2.1 Résultats de la simulation prospective - consommation par type de chauffage](#3.2.1prospective)
            * [3.2.2  Résultats de la simulation prospective - consommation par vecteur](#3.2.2prospective)
            * [3.2.2  Résultats de la simulation prospective - emissions](#3.2.3prospective)
    * [4. Biblioraphie](#biblio)
'''

app.layout = html.Div(children=[
    html.H1(children='Evolution of building heating consumption')]+
    [dbc.Row(dcc.Markdown(table_of_content))]+
    [dcc.Markdown('''# 2. Hypothèses du parc bâti'''),html.A(id="initialstate")]+
    [   dbc.Row(
            [dcc.Markdown(table_text[table]), html.A(id=anchor_id[table])]+
            [dbc.Col([dash_table.DataTable(id="table-editing-"+table,columns=columns_table[table],
                data=data_set_from_excel[table].round(3).to_dict('records'),editable=True)])]
        )
     for table in tables
    ]+
    [dbc.Row([html.Button(id='submit-button-state', n_clicks=0, children='Launch simulation')])]+
    [dcc.Markdown('''# 3.1 Etat actuel du parc'''),html.A(id="3.1actuel")] +
    [   dbc.Row(
            [dcc.Markdown(table_text[graph_name]), html.A(id=anchor_id[graph_name])]+
            [dbc.Col(dcc.Loading(children=[dcc.Graph(id=graph_name)]))]
        )
    for graph_name in Graphiques_Etat_actuel]+
    [dcc.Markdown('''# 3.2 Résultats de la simulation prospective'''),html.A(id="prospective")] +
       [dbc.Row(
           [dcc.Markdown(table_text[graph_name]), html.A(id=anchor_id[graph_name])] +
           [dbc.Col(dcc.Loading(children=[dcc.Graph(id='table-editing-simple-output_' + graph_name)]))]
       )
    for graph_name in Graphiques_simulation]
)

#https://github.com/robingirard/Energy-Alternatives-Planing/blob/master/Models/Prospective_conso/README.md
@app.callback(
    Output('output_actuel_conso','figure'),
    Output('output_actuel_emissions','figure'),
    Output('table-editing-simple-output_conso', 'figure'),
    Output('table-editing-simple-output_conso_finale', 'figure'),
    Output('table-editing-simple-output_emissions', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State("table-editing-"+"year_res_type", 'data'),
    State("table-editing-"+"year_res_type", 'columns')
)
def display_output(n_clicks,rows, columns):
    data_set_from_excel['year_res_type']=pd.DataFrame(rows, columns=[c['name'] for c in columns])
    sim_stock,sim_param,Energy_system_name = Compute_param_and_launch_simulation(data_set_from_excel)
    sim_stock_df = pd.concat(sim_stock, axis=0).reset_index(). \
        rename(columns={"level_0": "year"}).set_index(["year", "Energy_source", "building_type", "old_new"])

    col_class_dict = {'elec': 1, 'bois': 2, 'ordures': 2, 'autres': 2, 'gaz': 3, 'fioul': 3,
                      'charbon': 3}

    Var = "conso"
    all_columns = [Var + "_" + Vec for Vec in sim_param['Vecteurs']]
    y_df = sim_stock_df.loc[(2021, slice(None), slice(None))].reset_index().set_index(
        ['Energy_source', 'building_type', 'old_new']).filter(all_columns)
    y_df = pd.melt(y_df, value_name=Var, ignore_index=False)
    y_df[['v1', 'Vecteur']] = y_df['variable'].str.split('_', expand=True)
    y_df = y_df.drop(['v1', 'variable'], axis=1)
    y_df = y_df.reset_index().groupby(['building_type', 'Vecteur']).sum().reset_index()
    # y_df = y_df.loc[y_df["year"]==2021]
    y_df[Var] = y_df[Var] / 10 ** 9
    color_dict = gen_grouped_color_map(col_class_dict)
    y_df["class"] = [col_class_dict[cat] for cat in y_df["Vecteur"]]
    y_df = y_df.sort_values(by=['class'])

    locals()[Var] = y_df.copy()

    fig_actuel_conso = px.bar(y_df, x="building_type", y=Var, color="Vecteur", title="Wide-Form Input",
                 color_discrete_map=color_dict)
    fig_actuel_conso = fig_actuel_conso.update_layout(title_text="Consommation d'énergie finale par vecteur énergétique (en TWh)",
                            xaxis_title="Categorie",
                            yaxis_title="Consommation [TWh]")


    col_class_dict = {'elec': 1, 'bois': 2, 'ordures': 2, 'autres': 2, 'gaz': 3, 'fioul': 3,
                      'charbon': 3}
    Var = "emissions"
    all_columns = [Var + "_" + Vec for Vec in sim_param['Vecteurs']]
    y_df = sim_stock_df.loc[(2021, slice(None), slice(None))].reset_index().set_index(
        ['Energy_source', 'building_type', 'old_new']).filter(all_columns)
    y_df = pd.melt(y_df, value_name=Var, ignore_index=False)
    y_df[['v1', 'Vecteur']] = y_df['variable'].str.split('_', expand=True)
    y_df = y_df.drop(['v1', 'variable'], axis=1)
    y_df = y_df.reset_index().groupby(['building_type', 'Vecteur']).sum().reset_index()
    # y_df = y_df.loc[y_df["year"]==2021]
    y_df[Var] = y_df[Var] / 10 ** 12
    color_dict = gen_grouped_color_map(col_class_dict)
    y_df["class"] = [col_class_dict[cat] for cat in y_df["Vecteur"]]
    y_df = y_df.sort_values(by=['class'])

    locals()[Var] = y_df.copy()

    fig_actuel_emissions = px.bar(y_df, x="building_type", y=Var, color="Vecteur", title="Wide-Form Input",
                 color_discrete_map=color_dict)
    fig_actuel_emissions = fig_actuel_emissions.update_layout(title_text="Emissions de GES par vecteur énergétique (en MtCO2e)", xaxis_title="Categorie",
                            yaxis_title="Emissions [MtCO2e]")

    Var = "Conso"
    y_df = sim_stock_df.groupby(["year", Energy_system_name])[Var].sum().to_frame().reset_index(). \
        pivot(index=['year'], columns=Energy_system_name).loc[[year for year in sim_param["years"][1:]], Var]
    #y_df.columns = pd.MultiIndex.from_tuples([(str(col_class_dict[key]), key) for key in y_df.columns])
    fig_conso = MyStackedPlotly(y_df=y_df)
    fig_conso = fig_conso.update_layout(title_text="Conso énergie finale par mode de transport (en TWh)", xaxis_title="Année",
                            yaxis_title="Conso [TWh]")

    Var = "Conso"
    y_df = sim_stock_df.groupby(["year", "old_new"])[Var].sum().to_frame().reset_index(). \
               pivot(index=['year'], columns=['old_new']).loc[[year for year in sim_param["years"][1:]], Var] / 10 ** 9
    #y_df.columns = pd.MultiIndex.from_tuples([(str(col_class_dict[key]), key) for key in y_df.columns])
    fig_conso_finale = MyStackedPlotly(y_df=y_df)
    fig_conso_finale = fig_conso_finale.update_layout(title_text="Consommation d'énergie finale par mode chauffage (en TWh)", xaxis_title="Année",
                            yaxis_title="Conso [TWh]")
    Var = "emissions"
    y_df = sim_stock_df.groupby(["year", Energy_system_name])[Var].sum().to_frame().reset_index(). \
               pivot(index=['year'], columns=Energy_system_name).loc[
               [year for year in sim_param["years"][1:]], Var] / 10 ** 12
    #y_df.columns = pd.MultiIndex.from_tuples([(str(col_class_dict[key]), key) for key in y_df.columns])
    fig_emissions = MyStackedPlotly(y_df=y_df)
    fig_emissions = fig_emissions.update_layout(title_text="Emissions de GES par mode de chauffage (en MtCO2e)", xaxis_title="Année",
                            yaxis_title="Emissions [MtCO2e]")
    return fig_actuel_conso,fig_actuel_emissions,fig_conso,fig_conso_finale,fig_emissions
    #return {
    #    'data': [{
    #        'type': 'parcoords',
    #        'dimensions': [{
    #            'label': col['name'],
    #            'values': df[col['id']]
    #        } for col in columns]
    #    }]
    #}

if __name__ == '__main__':
 app.run_server(debug=True)