import matplotlib.pyplot as plt
import io
import base64

from zenml import step
from zenml.types import HTMLString


@step(enable_artifact_visualization=True)
def result_visualize(
    train_loss_list: list,
    train_acc_list: list,
) -> HTMLString:
    """Creates training loss & accuracy plots and shows them in ZenML dashboard."""

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss plot
    axes[0].plot(train_loss_list)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # Accuracy plot
    axes[1].plot(train_acc_list)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    # Convert plot to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Embed in HTML for ZenML dashboard
    html = f"""
    <div style="text-align:center;">
        <h3>Training Results</h3>
        <img src="data:image/png;base64,{image_base64}"
             style="max-width:100%; height:auto;" />
    </div>
    """

    return HTMLString(html)
