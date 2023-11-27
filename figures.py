import os
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from cnn import ConvNetModel
from dataset import BaseAudioDataset

def create_predictions_figure(
        state: dict,
        index: int,
        device: torch.device
) -> (go.Figure, str):
    model: ConvNetModel = state["model"]
    eval_set: BaseAudioDataset = state["eval_set"]
    categories = eval_set.get_categories()

    index %= len(eval_set)
    fig = make_subplots(
        rows=1,
        cols=2, # 1 plot for image, 1 for bar chart
        subplot_titles=[f"Sample {i+1}" for i in range(2)],
        horizontal_spacing=0.015,
        vertical_spacing=0.05
    )

    with torch.no_grad():
        model.eval()

        sample = eval_set[index]
        data, true_label, audio = (sample['spectrogram'], sample['metadata']['label'], sample['metadata']['path'])

        # Generate prediction and calculate loss
        # we 'unsqueeze' here to simulate having a batch size of 1,
        # as the model expects some batch dimension
        input = sample['spectrogram'].unsqueeze(0).to(device)
        output = model(input)

        pred_weights = torch.sigmoid_(output)
        topk_values, topk_indices = torch.topk(pred_weights.cpu().squeeze(0), 10)

        topk_values = topk_values.flip(dims=[0])
        topk_indices = topk_indices.flip(dims=[0])

        # Create subset of categories based on top-10 indices
        top_categories = [categories[i] for i in topk_indices.tolist()]

        # Add subplot
        fig.add_trace(go.Bar(y=top_categories, x=topk_values.numpy(), orientation='h', showlegend=False), col=1, row=1)

        fig.layout.annotations[0].update(text=f'Predictions', ax=0)
        fig.add_trace(go.Heatmap(z=data.cpu().squeeze(0).numpy(), colorscale='Viridis'), col=2, row=1)

        fig.layout.annotations[1].update(text=f'Label: {true_label}', ax=0)

        # the fraction of the window corresponding with the x-domain of this pair of charts
        leftmost = 0
        rightmost = 1
        predictions_domain = [leftmost, leftmost * 0.8 + rightmost * 0.2]
        spectrogram_domain = [leftmost * 0.75 + rightmost * 0.25, rightmost]

        fig.update_xaxes(domain=predictions_domain, row=1, col=1)
        fig.update_xaxes(domain=spectrogram_domain, row=1, col=2)

        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=50)
        )

    fig.update_layout(
        plot_bgcolor='rgba(0.3, 0.3, 0.3, 0.0)',
        paper_bgcolor='rgba(0.15, 0.15, 0.17, 1.0)',
        font=dict(color='rgba(0.5, 0.5, 0.5, 1.0)')
    )
    fig.update_coloraxes(showscale=False)
    audio_directory = state["audio_path"]
    # Remove the base directory part from the absolute path
    relative_path = os.path.relpath(audio, audio_directory)
    # Create the URL path
    audio_url = f'/audio/{relative_path}'

    return fig, audio_url


# def create_loss_figure(state: dict) -> (go.Figure, str):
#     fig = px.line(x=[0, 1, 2, 3, 4, 5, 6], y=[8, 6, 4, 3, 2, 1, 1])
#
#     fig.update(
#         margin=dict(l=10, r=10, t=10, b=10)
#     )
#     return fig