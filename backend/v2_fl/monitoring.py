# monitoring.py
import dash
from dash import dcc, html
import plotly.graph_objs as go
import json
import glob
import pandas as pd
import os
import threading
import time
from datetime import datetime, timedelta
import logging

class FederatedLearningMonitor:
    """
    Dashboard for monitoring federated learning progress.
    """
    
    def __init__(self, port=8050, metrics_dir='metrics'):
        self.port = port
        self.metrics_dir = metrics_dir
        self.app = dash.Dash(__name__, 
                            title="Federated Learning Monitor",
                            suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()
        self._running = False
        self._thread = None
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Federated Learning Monitor", style={'textAlign': 'center'}),
            
            # Tabs for different views
            dcc.Tabs([
                # Client Performance Tab
                dcc.Tab(label="Client Performance", children=[
                    html.Div([
                        html.H3("Learning Progress"),
                        dcc.Graph(id='accuracy-graph'),
                        dcc.Graph(id='loss-graph'),
                    ])
                ]),
                
                # Blockchain Transactions Tab
                dcc.Tab(label="Blockchain Transactions", children=[
                    html.Div([
                        html.H3("Transaction Metrics"),
                        dcc.Graph(id='gas-usage-graph'),
                        dcc.Graph(id='transaction-success-graph'),
                    ])
                ]),
                
                # System Health Tab
                dcc.Tab(label="System Health", children=[
                    html.Div([
                        html.H3("System Status"),
                        html.Div(id='ipfs-status'),
                        html.Div(id='blockchain-status'),
                        html.Div(id='active-clients'),
                    ])
                ]),
                
                # GA-Stacking Optimization Tab
                dcc.Tab(label="GA-Stacking", children=[
                    html.Div([
                        html.H3("GA-Stacking Progress"),
                        dcc.Graph(id='ga-fitness-graph'),
                        dcc.Graph(id='model-weights-graph'),
                    ])
                ]),
            ]),
            
            # Refresh interval
            dcc.Interval(id='interval-component', interval=5*1000), # 5 seconds
        ], style={'margin': '20px'})
    
    def _setup_callbacks(self):
        """Set up callbacks for updating dashboard components"""
        
        @self.app.callback(
            [dash.Output('accuracy-graph', 'figure'), 
            dash.Output('loss-graph', 'figure')],
            [dash.Input('interval-component', 'n_intervals')]
        )
        def update_performance_graphs(_):
            accuracy_data = []
            loss_data = []
            
            # Read client metrics files
            for client_dir in glob.glob(f'{self.metrics_dir}/client-*'):
                client_id = os.path.basename(client_dir)
                metrics_file = f'{client_dir}/metrics_history.json'
                
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_history = json.load(f)
                            
                            # Handle the list of dictionaries format
                            if isinstance(metrics_history, list):
                                # Extract rounds, accuracy and loss values
                                rounds = [entry.get('round', i) for i, entry in enumerate(metrics_history)]
                                accuracy_values = [entry.get('accuracy', 0) for entry in metrics_history]
                                
                                # Look for both fit_loss and eval_loss
                                loss_values = []
                                for entry in metrics_history:
                                    if 'fit_loss' in entry:
                                        loss_values.append(entry['fit_loss'])
                                    elif 'eval_loss' in entry:
                                        loss_values.append(entry['eval_loss'])
                                    elif 'loss' in entry:
                                        loss_values.append(entry['loss'])
                                    else:
                                        loss_values.append(0)
                                
                                # Create accuracy trace
                                accuracy_data.append(go.Scatter(
                                    x=rounds,
                                    y=accuracy_values,
                                    mode='lines+markers',
                                    name=f'{client_id} Accuracy'
                                ))
                                
                                # Create loss trace
                                loss_data.append(go.Scatter(
                                    x=rounds,
                                    y=loss_values,
                                    mode='lines+markers',
                                    name=f'{client_id} Loss'
                                ))
                            else:
                                # Handle the original format with separate arrays
                                if 'accuracy' in metrics_history and metrics_history['accuracy']:
                                    accuracy_data.append(go.Scatter(
                                        x=list(range(len(metrics_history['accuracy']))),
                                        y=metrics_history['accuracy'],
                                        mode='lines+markers',
                                        name=f'{client_id} Accuracy'
                                    ))
                                
                                if 'loss' in metrics_history and metrics_history['loss']:
                                    loss_data.append(go.Scatter(
                                        x=list(range(len(metrics_history['loss']))),
                                        y=metrics_history['loss'],
                                        mode='lines+markers',
                                        name=f'{client_id} Loss'
                                    ))
                    except Exception as e:
                        logging.error(f"Error reading metrics file {metrics_file}: {e}")
            
            accuracy_layout = {
                'title': 'Client Accuracy Over Rounds',
                'xaxis': {'title': 'Round'},
                'yaxis': {'title': 'Accuracy', 'range': [0, 100]},  # Adjust to handle percentages
                'legend': {'orientation': 'h', 'y': -0.2}
            }
            
            loss_layout = {
                'title': 'Client Loss Over Rounds',
                'xaxis': {'title': 'Round'},
                'yaxis': {'title': 'Loss'},
                'legend': {'orientation': 'h', 'y': -0.2}
            }
            
            return (
                {'data': accuracy_data, 'layout': accuracy_layout},
                {'data': loss_data, 'layout': loss_layout}
            )
            
        @self.app.callback(
            [dash.Output('ga-fitness-graph', 'figure'),
             dash.Output('model-weights-graph', 'figure')],
            [dash.Input('interval-component', 'n_intervals')]
        )
        def update_ga_graphs(_):
            # Process GA metrics
            ga_data = []
            weights_data = []
            
            for client_dir in glob.glob(f'{self.metrics_dir}/client-*'):
                client_id = os.path.basename(client_dir)
                ga_file = f'{client_dir}/ga_progress.json'
                
                if os.path.exists(ga_file):
                    try:
                        with open(ga_file, 'r') as f:
                            ga_metrics = json.load(f)
                            
                            # GA fitness over generations
                            if 'best_fitness' in ga_metrics:
                                ga_data.append(go.Scatter(
                                    x=list(range(len(ga_metrics['best_fitness']))),
                                    y=ga_metrics['best_fitness'],
                                    mode='lines',
                                    name=f'{client_id} Best Fitness'
                                ))
                            
                            if 'avg_fitness' in ga_metrics:
                                ga_data.append(go.Scatter(
                                    x=list(range(len(ga_metrics['avg_fitness']))),
                                    y=ga_metrics['avg_fitness'],
                                    mode='lines',
                                    name=f'{client_id} Avg Fitness',
                                    line={'dash': 'dot'}
                                ))
                            
                            # Model weights
                            if 'weights' in ga_metrics and ga_metrics['weights']:
                                final_weights = ga_metrics['weights'][-1]
                                model_names = list(final_weights.keys())
                                weights_values = list(final_weights.values())
                                
                                weights_data.append(go.Bar(
                                    x=model_names,
                                    y=weights_values,
                                    name=client_id
                                ))
                    except Exception as e:
                        logging.error(f"Error reading GA file {ga_file}: {e}")
            
            ga_layout = {
                'title': 'GA-Stacking Fitness Progression',
                'xaxis': {'title': 'Generation'},
                'yaxis': {'title': 'Fitness'},
                'legend': {'orientation': 'h', 'y': -0.2}
            }
            
            weights_layout = {
                'title': 'Final Model Weights by Client',
                'xaxis': {'title': 'Model Type'},
                'yaxis': {'title': 'Weight', 'range': [0, 1]},
                'barmode': 'group',
                'legend': {'orientation': 'h', 'y': -0.2}
            }
            
            return (
                {'data': ga_data, 'layout': ga_layout},
                {'data': weights_data, 'layout': weights_layout}
            )
        
        @self.app.callback(
            [dash.Output('gas-usage-graph', 'figure'),
            dash.Output('transaction-success-graph', 'figure')],
            [dash.Input('interval-component', 'n_intervals')]
        )
        def update_blockchain_graphs(_):
            # Read blockchain transaction metrics
            tx_file = f'{self.metrics_dir}/blockchain/transaction_metrics.jsonl'
            gas_data = []
            success_data = []
            
            if os.path.exists(tx_file):
                try:
                    # Parse JSONL file
                    transactions = []
                    with open(tx_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                transactions.append(json.loads(line))
                    
                    if transactions:
                        df = pd.DataFrame(transactions)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        
                        # For success rate by client and round
                        if 'client_id' in df.columns and 'success' in df.columns and 'round_num' in df.columns:
                            # Group by client_id and round_num
                            success_by_client_round = df.groupby(['client_id', 'round_num'])['success'].mean().reset_index()
                            
                            # Create a separate trace for each client
                            for client_id, client_df in success_by_client_round.groupby('client_id'):
                                success_data.append(go.Bar(
                                    x=client_df['round_num'],
                                    y=client_df['success'] * 100,  # Convert to percentage
                                    name=f'{client_id} Success Rate'
                                ))
                        
                        # Also add a trace for gas usage if available (even if null)
                        if 'round_num' in df.columns:
                            # Create dict mapping from round to count of transactions
                            tx_count_by_round = df.groupby('round_num').size().to_dict()
                            
                            rounds = sorted(tx_count_by_round.keys())
                            counts = [tx_count_by_round[r] for r in rounds]
                            
                            gas_data.append(go.Bar(
                                x=rounds,
                                y=counts,
                                name='Transactions Per Round'
                            ))
                        
                except Exception as e:
                    logging.error(f"Error reading transaction metrics: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
            
            gas_layout = {
                'title': 'Transactions Per Round',
                'xaxis': {'title': 'Round'},
                'yaxis': {'title': 'Number of Transactions'},
                'legend': {'orientation': 'h', 'y': -0.2}
            }
            
            success_layout = {
                'title': 'Transaction Success Rate by Client and Round',
                'xaxis': {'title': 'Round'},
                'yaxis': {'title': 'Success Rate (%)', 'range': [0, 101]},
                'barmode': 'group',
                'legend': {'orientation': 'h', 'y': -0.2}
            }
            
            return (
                {'data': gas_data, 'layout': gas_layout},
                {'data': success_data, 'layout': success_layout}
            )
        
        @self.app.callback(
            [dash.Output('ipfs-status', 'children'),
             dash.Output('blockchain-status', 'children'),
             dash.Output('active-clients', 'children')],
            [dash.Input('interval-component', 'n_intervals')]
        )
        def update_system_status(_):
            # Check active clients
            active_clients = len(glob.glob(f'{self.metrics_dir}/client-*'))
            
            # Read system status
            status_file = f'{self.metrics_dir}/system_status.json'
            ipfs_status = "Unknown"
            blockchain_status = "Unknown"
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                        ipfs_status = status.get('ipfs', {}).get('status', "Unknown")
                        blockchain_status = status.get('blockchain', {}).get('status', "Unknown")
                except Exception as e:
                    logging.error(f"Error reading system status: {e}")
            
            ipfs_color = "#4CAF50" if ipfs_status == "Connected" else "#F44336"
            blockchain_color = "#4CAF50" if blockchain_status == "Connected" else "#F44336"
            
            return (
                html.Div([
                    html.H4("IPFS Status"),
                    html.Div(ipfs_status, style={'color': ipfs_color, 'fontWeight': 'bold'})
                ]),
                html.Div([
                    html.H4("Blockchain Status"),
                    html.Div(blockchain_status, style={'color': blockchain_color, 'fontWeight': 'bold'})
                ]),
                html.Div([
                    html.H4("Active Clients"),
                    html.Div(str(active_clients), style={'fontWeight': 'bold'})
                ])
            )
    
    def start(self):
        """Start the monitoring server in a separate thread"""
        if self._running:
            logging.warning("Monitoring server is already running")
            return
        
        def run_server():
            # Use app.run() instead of app.run_server()
            self.app.run(debug=False, host='0.0.0.0', port=self.port)
        
        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        self._running = True
        logging.info(f"Monitoring server started at http://localhost:{self.port}")
    
    def stop(self):
        """Stop the monitoring server"""
        if not self._running:
            return
        
        self._running = False
        logging.info("Monitoring server stopped")


# Helper function to log system status for the dashboard
def log_system_status(ipfs_connected, blockchain_connected):
    """Log system status for the monitoring dashboard"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'ipfs': {
            'status': 'Connected' if ipfs_connected else 'Disconnected'
        },
        'blockchain': {
            'status': 'Connected' if blockchain_connected else 'Disconnected'
        }
    }
    
    # Ensure directory exists
    os.makedirs('metrics', exist_ok=True)
    
    # Write status to file
    with open('metrics/system_status.json', 'w') as f:
        json.dump(status, f)


# Helper function to log GA-Stacking progress
def log_ga_progress(client_id, generation, best_fitness, avg_fitness, weights):
    """Log GA-Stacking progress for the monitoring dashboard"""
    client_dir = f'metrics/{client_id}'
    os.makedirs(client_dir, exist_ok=True)
    
    ga_file = f'{client_dir}/ga_progress.json'
    
    # Read existing data if available
    ga_data = {
        'best_fitness': [],
        'avg_fitness': [],
        'weights': []
    }
    
    if os.path.exists(ga_file):
        try:
            with open(ga_file, 'r') as f:
                ga_data = json.load(f)
        except Exception:
            pass
    
    # Append new data
    ga_data['best_fitness'].append(best_fitness)
    ga_data['avg_fitness'].append(avg_fitness)
    ga_data['weights'].append(weights)
    
    # Write updated data
    with open(ga_file, 'w') as f:
        json.dump(ga_data, f)


# Function to start the monitoring server
def start_monitoring_server(port=8050):
    """Start the federated learning monitoring dashboard"""
    monitor = FederatedLearningMonitor(port=port)
    monitor.start()
    return monitor
