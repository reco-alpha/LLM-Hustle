import os
import torch

checkpoint_dir = 'model_check_points'

class Utils():
    # Function to load the latest checkpoint model or initialize from scratch
    # mode: pre-train | fine-tune
    def load_model(self, model_signature, mode):
        os.makedirs(checkpoint_dir, exist_ok=True)

        if(mode == 'pre-train'):
            os.makedirs(f"{checkpoint_dir}/pre-train", exist_ok=True)
            checkpoint_files = [f for f in os.listdir(f"{checkpoint_dir}/pre-train")]
            
            if checkpoint_files:
                latest_checkpoint =f"model_checkpoint_pre_train_{len(checkpoint_files)}.pth"
                checkpoint_path = os.path.join(f"{checkpoint_dir}/pre-train", latest_checkpoint)
                model = torch.load(checkpoint_path)
                print("Loaded 'Pre-train' checkpoint from:", latest_checkpoint)
            else:
                print("No 'Pre-train' checkpoints found, initializing model from scratch.")
                model = model_signature()
            
            return model
        elif(mode == 'fine-tune'):
            os.makedirs(f"{checkpoint_dir}/fine-tune", exist_ok=True)
            checkpoint_files = [f for f in os.listdir(f"{checkpoint_dir}/fine-tune")]
            
            if checkpoint_files:
                latest_checkpoint =f"model_checkpoint_fine_tune_{len(checkpoint_files)}.pth"
                checkpoint_path = os.path.join(f"{checkpoint_dir}/fine-tune", latest_checkpoint)
                model = torch.load(checkpoint_path)
                print("Loaded 'Fine-tune' checkpoint from:", latest_checkpoint)
            else:
                checkpoint_files = [f for f in os.listdir(f"{checkpoint_dir}/pre-train")]
                if checkpoint_files:
                    latest_checkpoint =f"model_checkpoint_pre_train_{len(checkpoint_files)}.pth"
                    checkpoint_path = os.path.join(f"{checkpoint_dir}/pre-train", latest_checkpoint)
                    model = torch.load(checkpoint_path)
                    print("Loaded 'Pre-train' checkpoint from:", latest_checkpoint)
                else:
                    print("No 'Fine-tune & Pre-train' checkpoints found, initializing model from scratch.")
                    model = model_signature()
            
            return model


    # Function to save the model
    # mode: pre-train | fine-tune
    def save_model(self, model, mode):
        os.makedirs(checkpoint_dir, exist_ok=True)

        if(mode == 'pre-train'):
            os.makedirs(f"{checkpoint_dir}/pre-train", exist_ok=True)
            checkpoint_files = [f for f in os.listdir(f"{checkpoint_dir}/pre-train") if f.endswith('.pth') and 'pre_train' in f]
            checkpoint_name = f"model_checkpoint_pre_train_{len(checkpoint_files) + 1}.pth"
            checkpoint_path = os.path.join(f"{checkpoint_dir}/pre-train", checkpoint_name)
            torch.save(model, checkpoint_path)
            print("Pre-train Model saved to:", checkpoint_name)
        elif(mode == 'fine-tune'):
            os.makedirs(f"{checkpoint_dir}/fine-tune", exist_ok=True)
            checkpoint_files = [f for f in os.listdir(f"{checkpoint_dir}/fine-tune") if f.endswith('.pth') and 'fine_tune' in f]
            checkpoint_name = f"model_checkpoint_fine_tune_{len(checkpoint_files) + 1}.pth"
            checkpoint_path = os.path.join(f"{checkpoint_dir}/fine-tune", checkpoint_name)
            torch.save(model, checkpoint_path)
            print("Fine-tune Model saved to:", checkpoint_name)
             



    def save_losses(self, losses_list):
        # Save accumulated losses to file
        losses_file = 'losses.txt'
        mode = 'a+' if os.path.exists(losses_file) else 'w+'

        with open(losses_file, mode) as f:
            if mode == 'w+':
                print("Creating a new losses file.")
            else:
                print("Loading past data from losses file.")

            for loss_item in losses_list:
                f.write(str(loss_item) + '\n')

        print("Losses saved to", losses_file)