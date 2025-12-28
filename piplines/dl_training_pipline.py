from zenml import pipeline
from steps.ingest_data import ingest_data_step
from steps.data_augumentation import data_augmentation_step
from steps.visualization import visualize
from steps.model_training import model_training_step
from steps.evaluation import evaluate_model_step
from steps.result import result_visualize as results_visualization_step
import torch
@pipeline
def dl_training_pipeline(data_path:str, username: str):
    """Defines the deep learning training pipeline."""
    model_path=rf"C:\Users\Amin\Desktop\MLopps\models\brain_tumor_cnn.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = ingest_data_step(data_path)
    visualize(data)
    train_loader, test_loader, class_names=data_augmentation_step(data)
    model, train_loss_list, train_acc_list , val_acc_list = model_training_step(username=username,model_path=model_path,train_loader=train_loader,device=device)
    accuracy, precision, recall = evaluate_model_step(model,test_loader,device,class_names)
    results_visualization_step(train_loss_list, train_acc_list)
    