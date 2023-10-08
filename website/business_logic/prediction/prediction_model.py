from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GANDALFConfig

class PredictionModel:
    
    ##BUILD MODEL
    def __init__(self, continuous_cols, categorical_cols, target):

        #Configuration setups
        
        #Data columns definition
        data_config = DataConfig(
            target=target,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )
    
        #Trainer configurations based on dataset size, and
        # experimental results
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
    
        #Configurations set based on 
        #Manu Joseph & Harsh Raj,
        #GANDALF: Gated Adaptive Network for Deep Automated Learning of Features
        model_config = GANDALFConfig(
            task= "classification", 
            gflu_stages= 8, 
            gflu_dropout= 0.041, 
            gflu_feature_init_sparsity= 0.35
        )
    
        #Initialize model with configuration
        self.tab_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

    #Method to make predictions with the trained model
    #The data parameter must not include the target column
    #Returns a pd.DataFrame
    def pred(self, data):
        return self.tab_model.predict(data)
    
    #Method to fit the model with train and validation data
    #Returns pl.Trainer: The PyTorch Lightning Trainer instance
    def fit_model(self, train, validation):
        #Fit method from PyTorch Tabular
        return self.tab_model.fit(train=train, validation=validation)
        
    #Method to evaluate the model with test data
    #Returns the final test result dictionary
    # Union[dict, list]
    def evaluate_model(self, test):
        #Evaluate method from PyTorch Tabular
        return self.tab_model.evaluate(test)
    
    #Method to return the feature importance of the model
    #Returns a pd.DataFrame
    def feature_importance(self):
        return self.tab_model.feature_importance()

    #Method to save the model in the following directory
    # "examples/basic"
    def save_model(self):
        self.tab_model.save_model("examples/basic")
