import os
import matplotlib.pyplot as plt
import io
import base64
import os
from zenml import step, pipeline
from zenml.types import HTMLString

@step(enable_artifact_visualization=True)
def visualize(datatype:str) -> HTMLString:
    """Creates a matplotlib plot and returns it as embedded HTML."""
    class_counts = {}
    for cls in os.listdir(datatype):
        class_path = os.path.join(datatype, cls)
        if os.path.isdir(class_path):
            class_counts[cls] = len(os.listdir(class_path))
    print("Image Count per Class:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    fig, ax = plt.subplots()
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of Images")
    ax.set_title("Image Count per Class")

    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Embed image in HTML
    html = f"""
    <div style='text-align:center;'>
        <img src='data:image/png;base64,{image_base64}' 
             style='max-width:100%; height:auto;'/>
    </div>
    """
    
    return HTMLString(html)