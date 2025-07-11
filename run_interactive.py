#!/usr/bin/env python
"""
Interactive Launcher for Gut Tissue Segmentation
Supports 3 models: UNet, HybridUNetGNN, Cellpose SAM
Features: Training, RL Human Feedback, Testing
Uses 15-class tissue configuration
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import subprocess
import threading

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Import simplified modules
from core.tissue_config import get_num_tissue_classes, get_all_tissue_names, get_all_tissue_colors
from core.visualization import create_class_legend
from core.data_handler import create_rl_data_loader, RLDataset

class InteractiveGUI:
    """Interactive GUI for gut tissue segmentation with training, RL, and testing"""
    
    def __init__(self, master):
        self.master = master
        self.setup_window()
        self.create_widgets()
        self.update_status("Ready - 15 tissue classes loaded", 'green')
    
    def setup_window(self):
        """Configure the main window"""
        self.master.title("Interactive Gut Tissue Segmentation")
        self.master.geometry("520x550")
        self.master.resizable(False, False)
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = tk.Frame(self.master, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Interactive Gut Tissue Segmentation", 
                             font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(main_frame, text="15 Tissue Classes • 3 Models • RL + Testing", 
                                font=('Arial', 10))
        subtitle_label.pack(pady=(0, 20))
        
        # Training section
        train_frame = tk.LabelFrame(main_frame, text="Training", padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(train_frame, text="Start Training", 
                 command=self.start_training, 
                 width=30, height=2, bg='lightgreen', font=('Arial', 11, 'bold')).pack(pady=5)
        
        # RL Human Feedback section
        rl_frame = tk.LabelFrame(main_frame, text="Reinforcement Learning", padx=10, pady=10)
        rl_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(rl_frame, text="Human Feedback RL", 
                 command=self.start_human_feedback_rl, 
                 width=30, height=2, bg='lightyellow', font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Testing section
        test_frame = tk.LabelFrame(main_frame, text="Testing", padx=10, pady=10)
        test_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(test_frame, text="Test Trained Model", 
                 command=self.start_model_testing, 
                 width=30, height=2, bg='lightcoral', font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Model info
        model_frame = tk.LabelFrame(main_frame, text="Supported Models", padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        models_text = "• UNet (Pretrained) - ResNet34 + ImageNet\n• HybridUNetGNN (Untrained) - UNet + GNN\n• Cellpose SAM (Pretrained) - Placeholder"
        tk.Label(model_frame, text=models_text, justify=tk.LEFT, font=('Arial', 9)).pack()
        
        # Tools section
        tools_frame = tk.LabelFrame(main_frame, text="Tools", padx=10, pady=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(tools_frame, text="View Tissue Classes", 
                 command=self.show_tissue_classes, 
                 width=25, bg='lightblue').pack(pady=2)
        
        tk.Button(tools_frame, text="Cleanup Temp Files", 
                 command=self.cleanup_temp_files, 
                 width=25, bg='lightyellow').pack(pady=2)
        
        # Status
        self.status_label = tk.Label(main_frame, text="Ready", 
                                   font=('Arial', 10), fg='black')
        self.status_label.pack(pady=(10, 0))
    
    def start_training(self):
        """Launch simple training with model selection"""
        try:
            print("DEBUG: start_training called")
            
            # Get data directory
            data_dir = filedialog.askdirectory(
                title="Select Data Directory (with images/ and masks/ folders)"
            )
            
            print(f"DEBUG: data_dir selected: {data_dir}")
            
            if not data_dir:
                self.update_status("No data directory selected", 'orange')
                return
            
            # Check data directory structure
            images_dir = os.path.join(data_dir, 'images')
            masks_dir = os.path.join(data_dir, 'masks')
            
            print(f"DEBUG: Checking directories - images: {os.path.exists(images_dir)}, masks: {os.path.exists(masks_dir)}")
            
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                messagebox.showerror(
                    "Invalid Data Directory",
                    f"Data directory must contain:\n"
                    f"• images/ folder\n"
                    f"• masks/ folder\n\n"
                    f"Found:\n"
                    f"• images/: {'✓' if os.path.exists(images_dir) else '✗'}\n"
                    f"• masks/: {'✓' if os.path.exists(masks_dir) else '✗'}"
                )
                return
            
            # Select model
            model_choice = self.select_model()
            print(f"DEBUG: model_choice selected: {model_choice}")
            
            if not model_choice:
                return
            
            # Get training parameters
            epochs = simpledialog.askinteger("Training Epochs", 
                                           "Number of training epochs:", 
                                           initialvalue=10, minvalue=1, maxvalue=100)
            print(f"DEBUG: epochs selected: {epochs}")
            
            if not epochs:
                return
            
            batch_size = simpledialog.askinteger("Batch Size", 
                                               "Batch size for training:", 
                                               initialvalue=4, minvalue=1, maxvalue=16)
            print(f"DEBUG: batch_size selected: {batch_size}")
            
            if not batch_size:
                return
            
            # Use pretrained weights?
            use_pretrained = messagebox.askyesno("Pretrained Weights", 
                                               "Use pretrained weights (where available)?")
            print(f"DEBUG: use_pretrained selected: {use_pretrained}")
            
            # Build command
            cmd = [
                "python", "training/train_comprehensive.py",
                "--model_type", model_choice,
                "--data_dir", data_dir,
                "--num_epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--save_dir", "training_output"
            ]
            
            if use_pretrained:
                # Add pretrained model paths based on model type
                if model_choice == "unet":
                    pretrained_path = "pretrained_models/unet_medical.pth"
                    if os.path.exists(pretrained_path):
                        cmd.extend(["--pretrained_path", pretrained_path])
                elif model_choice == "cellpose_sam":
                    pretrained_path = "pretrained_models/cellpose_sam.pth"
                    if os.path.exists(pretrained_path):
                        cmd.extend(["--pretrained_path", pretrained_path])
            
            print(f"DEBUG: Built command: {cmd}")
            
            # Show confirmation
            confirm_msg = (
                f"Ready to start training!\n\n"
                f"Model: {model_choice}\n"
                f"Data: {os.path.basename(data_dir)}\n"
                f"Epochs: {epochs}\n"
                f"Batch Size: {batch_size}\n"
                f"Pretrained: {'Yes' if use_pretrained else 'No'}\n\n"
                f"Continue?"
            )
            
            if not messagebox.askyesno("Confirm Training", confirm_msg):
                print("DEBUG: User cancelled confirmation")
                return
            
            print("DEBUG: About to launch subprocess")
            
            # Launch training
            self.update_status(f"Starting {model_choice} training...", 'blue')
            
            if sys.platform == "win32":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd)
            
            print("DEBUG: Subprocess launched successfully")
            
            self.update_status(f"Training launched: {model_choice}", 'green')
            
            messagebox.showinfo(
                "Training Started",
                f"Training has been launched!\n\n"
                f"Check the terminal window for progress.\n"
                f"Models will be saved to 'training_output/' directory.\n\n"
                f"Training may take several minutes to hours\n"
                f"depending on your data and hardware."
            )
            
        except Exception as e:
            print(f"DEBUG: Exception occurred: {e}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            messagebox.showerror("Training Error", f"Failed to start training:\n{str(e)}")
            self.update_status("Training failed", 'red')
    
    def start_human_feedback_rl(self):
        """Launch Human Feedback Reinforcement Learning"""
        try:
            self.update_status("Starting Human Feedback RL...", 'blue')
            
            # Show RL information
            info_msg = (
                "Human Feedback Reinforcement Learning\n\n"
                "This system allows you to:\n"
                "• Load existing trained models\n"
                "• Provide feedback on predictions\n"
                "• Improve model performance iteratively\n"
                "• Use your existing RL dataset infrastructure\n\n"
                "Requirements:\n"
                "• At least one trained model (.pth file)\n"
                "• Images for RL feedback (no masks needed)\n\n"
                "Continue with RL setup?"
            )
            
            if not messagebox.askyesno("Human Feedback RL", info_msg):
                self.update_status("RL cancelled", 'orange')
                return
            
            # Select trained model
            model_path = self.select_trained_model()
            if not model_path:
                return
            
            # Select RL data directory
            rl_data_dir = filedialog.askdirectory(
                title="Select RL Data Directory (images for feedback)"
            )
            
            if not rl_data_dir:
                self.update_status("No RL data directory selected", 'orange')
                return
            
            # Launch RL system
            self.launch_rl_system(model_path, rl_data_dir)
                
        except Exception as e:
            messagebox.showerror("RL Error", f"Failed to start Human Feedback RL:\n{str(e)}")
            self.update_status("RL failed", 'red')
    
    def launch_rl_system(self, model_path, data_dir):
        """Launch the RL system with simple interface"""
        try:
            # Create simple RL script call
            cmd = [
                "python", "-c", f"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from data_handler import create_rl_data_loader, RLDataset
import torch

print("Human Feedback RL System")
print(f"Model: {os.path.basename('{model_path}')}") 
print(f"Data: {os.path.basename('{data_dir}')}")
print()

# Create RL dataset
try:
    rl_loader, rl_dataset = create_rl_data_loader('{data_dir}', batch_size=1, shuffle=False)
    print(f"RL Dataset created: {{len(rl_dataset)}} images")
    print()
    print("RL System ready for human feedback!")
    print("This is a placeholder - implement your specific RL feedback interface here.")
    print()
    print("Next steps:")
    print("1. Load model from: {model_path}")
    print("2. Show predictions to user")
    print("3. Collect feedback")
    print("4. Update model based on feedback")
    print()
    input("Press Enter to continue...")
                
        except Exception as e:
    print(f"Error setting up RL: {{e}}")
    input("Press Enter to continue...")
"""
            ]
            
            self.update_status("Launching RL system...", 'blue')
            
            if sys.platform == "win32":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd)
            
            self.update_status("RL system launched", 'green')
            
            messagebox.showinfo(
                "RL System Launched",
                f"Human Feedback RL has been launched!\n\n"
                f"Model: {os.path.basename(model_path)}\n"
                f"Data: {os.path.basename(data_dir)}\n\n"
                f"Check the terminal window for the RL interface.\n"
                f"This is a simple placeholder - customize as needed."
            )
                
        except Exception as e:
            messagebox.showerror("RL Launch Error", f"Failed to launch RL system:\n{str(e)}")
            self.update_status("RL launch failed", 'red')
    
    def start_model_testing(self):
        """Launch model testing interface"""
        try:
            self.update_status("Starting model testing...", 'blue')
            
            # Show testing information
            info_msg = (
                "Model Testing System\n\n"
                "Test your trained models with:\n"
                "• Accuracy metrics calculation\n"
                "• Prediction visualization\n"
                "• Performance analysis\n"
                "• Tissue class predictions\n\n"
                "Requirements:\n"
                "• Trained model (.pth file)\n"
                "• Test images (with or without masks)\n\n"
                "Continue with testing setup?"
            )
            
            if not messagebox.askyesno("Model Testing", info_msg):
                self.update_status("Testing cancelled", 'orange')
                return
            
            # Select trained model
            model_path = self.select_trained_model()
            if not model_path:
                return
            
            # Select test data
            test_data_dir = filedialog.askdirectory(
                title="Select Test Data Directory"
            )
            
            if not test_data_dir:
                self.update_status("No test data directory selected", 'orange')
                return
            
            # Launch testing system
            self.launch_testing_system(model_path, test_data_dir)
            
        except Exception as e:
            messagebox.showerror("Testing Error", f"Failed to start model testing:\n{str(e)}")
            self.update_status("Testing failed", 'red')
    
    def launch_testing_system(self, model_path, test_dir):
        """Launch the testing system"""
        try:
            # Create simple testing script call
            cmd = [
                "python", "-c", f"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

import torch
from data_handler import create_data_loaders
from tissue_config import get_num_tissue_classes

print("Model Testing System")
print(f"Model: {os.path.basename('{model_path}')}") 
print(f"Test Data: {os.path.basename('{test_dir}')}")
print(f"Tissue Classes: {{get_num_tissue_classes()}}")
print()

# Check if test directory has proper structure
images_dir = os.path.join('{test_dir}', 'images')
masks_dir = os.path.join('{test_dir}', 'masks')

has_images = os.path.exists(images_dir)
has_masks = os.path.exists(masks_dir)

print(f"Test directory structure:")
print(f"• images/: {{'Yes' if has_images else 'No'}}")
print(f"• masks/: {{'Yes' if has_masks else 'No'}}")
print()

if has_images:
    print("Testing system ready!")
    print()
    print("Available test modes:")
    print("1. Prediction only (no accuracy metrics)")
    print("2. Full evaluation (with accuracy metrics)")
    print()
    print("This is a placeholder - implement your specific testing interface here.")
    print()
    print("Next steps:")
    print("1. Load model from: {model_path}")
    print("2. Process test images")
    print("3. Generate predictions")
    print("4. Calculate metrics (if masks available)")
    print("5. Save results")
else:
    print("No 'images' directory found in test data")
print()
input("Press Enter to continue...")
"""
            ]
            
            self.update_status("Launching testing system...", 'blue')
            
            if sys.platform == "win32":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd)
            
            self.update_status("Testing system launched", 'green')
            
            messagebox.showinfo(
                "Testing System Launched",
                f"Model testing has been launched!\n\n"
                f"Model: {os.path.basename(model_path)}\n"
                f"Test Data: {os.path.basename(test_dir)}\n\n"
                f"Check the terminal window for testing interface.\n"
                f"This is a simple placeholder - customize as needed."
            )
            
        except Exception as e:
            messagebox.showerror("Testing Launch Error", f"Failed to launch testing system:\n{str(e)}")
            self.update_status("Testing launch failed", 'red')
    
    def select_trained_model(self):
        """Select a trained model file"""
        # Look for existing models
        model_dirs = ["training_output", "models", "output"]
        model_files = []
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            model_files.append(os.path.join(root, file))
        
        if model_files:
            suggestion = f"Found {len(model_files)} trained model(s):\n"
            suggestion += "\n".join([f"• {os.path.basename(f)}" for f in model_files[:5]])
            if len(model_files) > 5:
                suggestion += f"\n• ... and {len(model_files) - 5} more"
            messagebox.showinfo("Available Models", suggestion)
        
        model_path = filedialog.askopenfilename(
            title="Select Trained Model (.pth file)",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
            initialdir="training_output" if os.path.exists("training_output") else "."
        )
        
        if not model_path:
            self.update_status("No model selected", 'orange')
        
        return model_path
    
    def select_model(self):
        """Model selection dialog for training"""
        model_window = tk.Toplevel(self.master)
        model_window.title("Select Model")
        model_window.geometry("400x300")
        model_window.grab_set()  # Make modal
        
        selected_model = tk.StringVar()
        
        tk.Label(model_window, text="Choose Training Model", 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Model options
        models = [
            ("unet", "UNet (Pretrained)", "ResNet34 backbone + ImageNet weights\nFast training, good performance"),
            ("hybrid_unet_gnn", "HybridUNetGNN (Untrained)", "UNet + Graph Neural Network\nAdvanced but requires more training"),
            ("cellpose_sam", "Cellpose SAM (Pretrained)", "Cellpose + Segment Anything\nExperimental - uses UNet fallback")
        ]
        
        for model_id, model_name, description in models:
            frame = tk.Frame(model_window)
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Radiobutton(frame, text=model_name, variable=selected_model, 
                          value=model_id, font=('Arial', 11, 'bold')).pack(anchor=tk.W)
            tk.Label(frame, text=description, font=('Arial', 9), 
                    fg='gray').pack(anchor=tk.W, padx=20)
        
        # Buttons
        button_frame = tk.Frame(model_window)
        button_frame.pack(pady=20)
        
        def confirm():
            model_window.destroy()
        
        def cancel():
            selected_model.set("")
            model_window.destroy()
        
        tk.Button(button_frame, text="Select", command=confirm, 
                 bg='lightgreen', width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel, 
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # Set default
        selected_model.set("unet")
        
        # Wait for window to close
        model_window.wait_window()
        
        return selected_model.get()
    
    def show_tissue_classes(self):
        """Display tissue class information"""
        try:
            num_classes = get_num_tissue_classes()
            tissue_names = get_all_tissue_names()
            tissue_colors = get_all_tissue_colors()
            
            # Create info window
            info_window = tk.Toplevel(self.master)
            info_window.title("Tissue Classes")
            info_window.geometry("500x600")
            
            # Title
            tk.Label(info_window, text=f"{num_classes} Tissue Classes", 
                    font=('Arial', 16, 'bold')).pack(pady=10)
            
            # Create scrollable frame
            canvas = tk.Canvas(info_window)
            scrollbar = tk.Scrollbar(info_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Add tissue class info
            for class_id in range(num_classes):
                name = tissue_names.get(class_id, f"Class_{class_id}")
                color = tissue_colors.get(class_id, (128, 128, 128))
                
                frame = tk.Frame(scrollable_frame)
                frame.pack(fill=tk.X, padx=10, pady=2)
                
                # Color box
                color_frame = tk.Frame(frame, width=30, height=20, 
                                     bg=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
                color_frame.pack(side=tk.LEFT, padx=(0, 10))
                color_frame.pack_propagate(False)
                
                # Class info
                tk.Label(frame, text=f"{class_id}: {name}", 
                        font=('Arial', 10)).pack(side=tk.LEFT, anchor=tk.W)
                tk.Label(frame, text=f"RGB{color}", 
                        font=('Arial', 8), fg='gray').pack(side=tk.RIGHT)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Generate legend button
            tk.Button(info_window, text="Generate Legend Image", 
                     command=lambda: self.generate_legend_image(info_window), 
                     bg='lightblue').pack(pady=10)
            
            self.update_status("Tissue classes displayed", 'green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show tissue classes:\n{str(e)}")
    
    def generate_legend_image(self, parent_window):
        """Generate and save tissue class legend image"""
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Legend Image",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfilename="tissue_legend.png"
            )
            
            if save_path:
                create_class_legend(save_path)
                messagebox.showinfo("Success", f"Legend saved to:\n{save_path}")
                self.update_status("Legend image generated", 'green')
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate legend:\n{str(e)}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            from core.data_handler import TempConvertedCleaner
            
            info = TempConvertedCleaner.get_temp_dir_info()
            
            if "No temp directories" in info:
                messagebox.showinfo("Cleanup", "No temporary files to clean!")
                self.update_status("No temp files found", 'green')
                return
            
            if messagebox.askyesno("Cleanup Confirmation", 
                                 f"Clean up temporary files?\n\n{info}"):
                TempConvertedCleaner.cleanup_all_temp_dirs()
                messagebox.showinfo("Cleanup Complete", "Temporary files cleaned!")
                self.update_status("Cleanup completed", 'green')
                
        except Exception as e:
            messagebox.showerror("Cleanup Error", f"Failed to cleanup:\n{str(e)}")
            self.update_status("Cleanup failed", 'red')
    
    def update_status(self, message, color='black'):
        """Update status label"""
        self.status_label.config(text=message, fg=color)
        self.master.update()

def main():
    """Main entry point"""
    print("Interactive Gut Tissue Segmentation")
    print(f"Tissue classes configured: {get_num_tissue_classes()}")
    print("Starting Interactive GUI...")
    
    root = tk.Tk()
    app = InteractiveGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()