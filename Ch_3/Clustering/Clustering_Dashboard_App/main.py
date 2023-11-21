import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px


# Load the data
merged_df = pd.read_csv("merged_df_stored_for_dash/"
                        "merged_df_for_dash_app_15_500.csv")
merged_df.drop(columns=["Unnamed: 0", "Acetate (mM)", "Boric Acid (mM)"], inplace=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
app.layout = dbc.Container(
    [
        # Title
        html.H2("Nucleic Acid Origami Literature Experiments MobileNetV2 Extracted AFM Image Features"
                " clustered with PCA followed by t-SNE",
                style={"text-align": "center"}),
        html.Div(
            [
                html.A(
                    html.Img(src="assets/github-mark.png", alt="View on GitHub",
                             style={"width": "30px", "height": "30px"}),
                    href="https://github.com/JordanConnolly/AI-strategies-for-Nucleic-Acid-Origami-experiments/",
                    target="_blank",
                    style={"text-align": "center", "display": "block"},
                )
            ],
            style={"margin-bottom": "20px"}
        ),
        dcc.Store(id="selected-data-store", data=None),
        dcc.Tabs(id="tabs", value="tab1", children=[
            dcc.Tab(label="Scatter Plot", value="tab1"),
            dcc.Tab(label="Correlation Plot", value="tab2")
        ]),
        html.Div(id="tabs-content"),
    ],
    fluid=True
)


@app.callback(Output("tabs-content", "children"),
              Input("tabs", "value"))
def render_content(tab):
    if tab == "tab1":
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id='hue-dropdown',
                                    options=[{'label': col, 'value': col} for col in merged_df.columns],
                                    value='Magnesium (mM)',
                                    placeholder="Select a variable for hue",
                                ),
                            ],
                            md=6
                        ),
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id='palette-dropdown',
                                    options=[
                                        {'label': 'Viridis', 'value': 'viridis'},
                                        {'label': 'Plasma', 'value': 'plasma'},
                                        {'label': 'Inferno', 'value': 'inferno'},
                                        {'label': 'Magma', 'value': 'magma'},
                                        {'label': 'Cividis', 'value': 'cividis'},
                                        {'label': 'Coolwarm', 'value': 'RdBu'},
                                    ],
                                    value='viridis',
                                    placeholder="Select a color palette",
                                ),
                            ],
                            md=6
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="scatter-plot", config={"displayModeBar": False, "scrollZoom": False}),
                            ],
                            md=12
                        ),
                        dbc.Col(
                            [
                                dcc.Store(id='drag-mode', data='select'),
                                dcc.Interval(id='drag-mode-interval', interval=1000, n_intervals=0),
                                dbc.Table(id="table", striped=True, bordered=True, hover=True),
                            ]
                        )
                    ]
                ),
            ],
            fluid=True
        )
    elif tab == "tab2":
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="scatter-plot-tab2",
                                          config={"displayModeBar": False, "scrollZoom": False}),
                            ],
                            md=12
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id='correlation-method-dropdown',
                                    options=[
                                        {'label': 'Use Pearson Correlation', 'value': 'pearson'},
                                        {'label': 'Use Spearman Correlation', 'value': 'spearman'},
                                        {'label': 'Use Kendall Correlation', 'value': 'kendall'},
                                    ],
                                    value='pearson',
                                    placeholder="Select a correlation method",
                                ),
                            ],
                            md=12
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="correlation-plot", config={"displayModeBar": False, "scrollZoom": False},
                                          style={'height': '1000px', 'width': '100%'})
                            ],
                            md=12
                        )
                    ]
                ),
            ],
            fluid=True
        )


@app.callback(
    Output("scatter-plot", "figure"),
    State("scatter-plot", "selectedData"),
    State("scatter-plot", "clickData"),
    Input("hue-dropdown", "value"),
    Input("palette-dropdown", "value"),
)
def update_scatter_plot(selectedData, clickData, hue_value, palette_value):
    fig = px.scatter(merged_df, x='cluster1', y='cluster2', color=hue_value,
                     color_continuous_scale=palette_value,
                     hover_data=['Paper Number', 'Experiment Number', 'Scaffold Name',
                                 'Scaffold Length (bases)', 'Structure Dimension',
                                 'Yield (%)', 'Yield Range (%)', 'Characterised By',
                                 'Scaffold to Staple Ratio', 'Constructed By', 'Buffer Name',
                                 'MgCl2 Used', 'Magnesium Acetate Used', 'Peak Temperature (oC)',
                                 'Base Temperature (oC)', 'Scaffold Molarity (nM)',
                                 'nanostructure length (nm)', 'nanostructure width (nm)',
                                 'number of individual staples', 'overall buffer pH', 'TRIS-HCl (mM)',
                                 'NaCl (mM)', 'Acetic acid (mM)',
                                 'EDTA (mM)', 'Temperature Ramp (s)'])

    fig.update_layout(
        height=800,  # Adjust the height of the graph
        title_x=0.5,
        xaxis_title="cluster1",
        yaxis_title="cluster2",
        dragmode="lasso"  # Set dragmode to "select"
    )

    # Check if points are selected
    if selectedData:
        # If points are selected, set the `selected` property to the selected points
        selected_points = [point['pointIndex'] for point in selectedData['points']]
        fig.update_traces(selectedpoints=selected_points)
    else:
        # If no points are selected, clear the `selected` property
        fig.update_traces(selectedpoints=None)

    return fig


@app.callback(
    Output("scatter-plot-tab2", "figure"),
    Input("tabs", "value"),
    State("scatter-plot-tab2", "selectedData"),
    State("scatter-plot-tab2", "clickData"),
)
def update_scatter_plot_tab2(tab, selectedData, clickData):
    if tab == "tab2":
        fig = px.scatter(merged_df, x='cluster1', y='cluster2', color='Magnesium (mM)',
                         hover_data=['Paper Number', 'Experiment Number', 'Scaffold Name',
                                     'Scaffold Length (bases)', 'Structure Dimension',
                                     'Yield (%)', 'Yield Range (%)', 'Characterised By',
                                     'Scaffold to Staple Ratio', 'Constructed By', 'Buffer Name',
                                     'MgCl2 Used', 'Magnesium Acetate Used', 'Peak Temperature (oC)',
                                     'Base Temperature (oC)', 'Scaffold Molarity (nM)',
                                     'nanostructure length (nm)', 'nanostructure width (nm)',
                                     'number of individual staples', 'overall buffer pH', 'TRIS-HCl (mM)',
                                     'NaCl (mM)', 'Acetic acid (mM)',
                                     'EDTA (mM)', 'Temperature Ramp (s)'])

        fig.update_layout(
            height=800,  # Adjust the height of the graph
            title_x=0.5,
            xaxis_title="cluster1",
            yaxis_title="cluster2",
            dragmode="lasso"  # Set dragmode to "select"
        )

        # Check if points are selected
        if selectedData:
            # If points are selected, set the `selected` property to the selected points
            selected_points = [point['pointIndex'] for point in selectedData['points']]
            fig.update_traces(selectedpoints=selected_points)
        else:
            # If no points are selected, clear the `selected` property
            fig.update_traces(selectedpoints=None)

        return fig
    else:
        return {}


@app.callback(
    Output('table', 'children'),
    [Input('scatter-plot', 'selectedData')],
)
def update_table(selectedData):
    if selectedData:
        # Get selected points
        points = selectedData['points']

        # Get data corresponding to selected points
        selected_indices = [point['pointIndex'] for point in points]
        selected_df = merged_df.iloc[selected_indices]

        # Update table with selected data
        return dbc.Col(
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in selected_df.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(selected_df.iloc[i][col]) for col in selected_df.columns
                    ]) for i in range(len(selected_df))
                ])
            ]),
            md=12,
        )
    else:
        return None


@app.callback(
    Output('correlation-plot', 'figure'),
    [Input('tabs', 'value'),
     Input('scatter-plot-tab2', 'selectedData'),
     Input('correlation-method-dropdown', 'value')])
def update_correlation_plot(tab, selected_data, correlation_method):
    global corr_matrix
    if tab == "tab2":
        if selected_data and selected_data != "all":
            selected_points = [point['pointIndex'] for point in selected_data['points']] if selected_data else []
            selected_df = merged_df.iloc[selected_points]
        elif selected_data == "all":
            selected_df = merged_df
        else:
            selected_df = pd.DataFrame()

        if not selected_df.empty:
            if correlation_method == 'pearson':
                corr_matrix = selected_df.corr(method='pearson')
            elif correlation_method == 'spearman':
                corr_matrix = selected_df.corr(method='spearman')
            elif correlation_method == 'kendall':
                corr_matrix = selected_df.corr(method='kendall')

            fig = px.imshow(corr_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1)
            title = "Correlation Plot for Selected Points"
        else:
            empty_corr_matrix = pd.DataFrame(0, index=merged_df.columns, columns=merged_df.columns)
            fig = px.imshow(empty_corr_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1)
            title = "Correlation Plot (No Points Selected)"

        fig.update_layout(
            title=title,
            title_x=0.5,
            margin=dict(l=0, r=0, t=30, b=0),
            height=600,
            coloraxis_colorbar=dict(title="Correlation")
        )

        return fig
    else:
        return {}


if __name__ == '__main__':
    app.run_server(debug=True)
