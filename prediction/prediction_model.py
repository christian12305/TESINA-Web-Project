from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GANDALFConfig

class PredictionModel:
    
    ##BUILD MODEL
    def __init__(self, classification_data, continuous_cols, categorical_cols, validation_data, target):

        #Configuration setups

        data_config = DataConfig(
            target=target,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )
    
        trainer_config = TrainerConfig(
            # Runs the LRFinder to automatically derive a learning rate
            auto_lr_find=True, 
            batch_size=32,
            max_epochs=100,
            accelerator="cpu",
            # Save best checkpoint monitoring val_loss
            checkpoints="valid_loss", 
            # After training, load the best checkpoint
            load_best=True, 
            #early_stopping=None,
            #checkpoints=None,
            #fast_dev_run=True,
        )
    
        optimizer_config = OptimizerConfig()
    
        model_config = GANDALFConfig(
            task= "classification", 
            gflu_stages= 8, 
            gflu_dropout= 0.041, 
            gflu_feature_init_sparsity= 0.35
        )
    
        #Initialize model with configuration
        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )            
    
    #Method to fit the model with train and validation data
    #Returns pl.Trainer: The PyTorch Lightning Trainer instance
    def fit_model(self, train, validation):
        #Fit method from PyTorch Tabular
        return self.tabular_model.fit(train=train, validation=validation)
        

    #Method to evaluate the model with test data
    #Returns the final test result dictionary
    # Union[dict, list]
    def evaluate_model(self, test):
        #Evaluate method from PyTorch Tabular
        return self.tabular_model.evaluate(test)

        
    #Method to make predictions with the trained model
    #The data parameter must not include the target column
    #Returns a pd.DataFrame
    def predict(self, data):
        return self.tabular_model.predict(data)
    

    #Method to return the feature importance of the model
    #Returns a pd.DataFrame
    def feature_importance(self):
        return self.tabular_model.feature_importance()

    #Method to save the model in the following directory
    # "examples/basic"
    def save_model(self):
        self.tabular_model.save_model("examples/basic")