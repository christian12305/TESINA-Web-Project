import inspect
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torchmetrics
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.model_summary import summarize
from rich.progress import track
from sklearn.base import TransformerMixin
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.utils import get_logger, pl_load
from pytorch_tabular.models.common.layers import Add, Embedding1dLayer, GatedFeatureLearningUnit
from pytorch_tabular.models.common.layers.activations import t_softmax
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from pytorch_tabular.models.common.heads import LinearHead
from .config import PredictionModelConfig
from omegaconf import OmegaConf
from pytorch_tabular.config.config import InferredConfig
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


logger = get_logger(__name__)
    
class GANDALFBackbone(nn.Module):
    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = True,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = self.n_continuous_features + self._embedded_cat_features
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()

    def _build_network(self):
        self.gflus = GatedFeatureLearningUnit(
            n_features_in=self.n_features,
            n_stages=self.gflu_stages,
            feature_mask_function=t_softmax,
            dropout=self.gflu_dropout,
            feature_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
        )

    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.n_continuous_features,
            categorical_embedding_dims=self.cat_embedding_dims,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gflus(x)

    @property
    def feature_importance_(self):
        return self.gflus.feature_mask_function(self.gflus.feature_masks).sum(dim=0).detach().cpu().numpy()

class PredictionModel(pl.LightningModule):
    def __init__(
            self,
            categorical_cols,
            continuous_cols,
            gflu_dropout,
            gflu_stages,
            gflu_feature_init_sparsity,
            batch_size,
            max_epochs,
            target,
            learning_rate: Optional[float] = 1e-3,
            # learning_rate: Optional[float] = 0.02089296130854041,
            auto_lr_find: Optional[bool] = True,
            config: Optional[Union[PredictionModelConfig, str]] = None,
            custom_loss: Optional[torch.nn.Module] = None,
            custom_metrics: Optional[List[Callable]] = None,
            custom_optimizer: Optional[torch.optim.Optimizer] = None,
            custom_optimizer_params: Dict = {},
        ):

        # super(PredictionModel,self).__init__()
        super().__init__()
        cont_dim = len(continuous_cols)
        cat_dim = len(categorical_cols)

        if config is None:
        # every default config needed for the model will be contained here
            config = PredictionModelConfig(
                categorical_cols = categorical_cols, 
                continuous_cols = continuous_cols, 
                categorical_dim = cat_dim,
                continuous_dim = cont_dim,
                auto_lr_find = auto_lr_find,
                learning_rate = learning_rate,
                gflu_dropout = gflu_dropout,
                gflu_stages = gflu_stages,
                gflu_feature_init_sparsity = gflu_feature_init_sparsity,
                batch_size = batch_size,
                max_epochs = max_epochs,
                target = target
            )
        
        self.config = self._read_parse_config(config, PredictionModelConfig)
        self.custom_loss = custom_loss
        self.custom_metrics = custom_metrics
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params
        self.learning_rate = learning_rate

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        # Backbone
        self._backbone = GANDALFBackbone(
            cat_embedding_dims=self.config.embedding_dims,
            n_continuous_features=self.config.continuous_dim,
            gflu_stages=self.config.gflu_stages,
            gflu_dropout=self.config.gflu_dropout,
            gflu_feature_init_sparsity=self.config.gflu_feature_init_sparsity,
            learnable_sparsity=self.config.learnable_sparsity,
            batch_norm_continuous_input=self.config.batch_norm_continuous_input,
            embedding_dropout=self.config.embedding_dropout,
        )
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self.T0 = nn.Parameter(torch.rand(self.config.output_dim), requires_grad=True)
        self._head = nn.Sequential(self._get_head_from_config(), Add(self.T0))

    def feature_importance(self) -> pd.DataFrame:
        """Returns a dataframe with feature importance for the model."""
        if hasattr(self.backbone, "feature_importance_"):
            imp = self.backbone.feature_importance_
            n_feat = len(self.config.categorical_cols + self.config.continuous_cols)
            if self.config.categorical_dim > 0:
                if imp.shape[0] != n_feat:
                    # Combining Cat Embedded Dimensions to a single one by averaging
                    wt = []
                    norm = []
                    ft_idx = 0
                    for _, embd_dim in self.config.embedding_dims:
                        wt.extend([ft_idx] * embd_dim)
                        norm.append(embd_dim)
                        ft_idx += 1
                    for _ in self.config.continuous_cols:
                        wt.extend([ft_idx])
                        norm.append(1)
                        ft_idx += 1
                    imp = np.bincount(wt, weights=imp) / np.array(norm)
                else:
                    # For models like FTTransformer, we dont need to do anything
                    # It takes categorical and continuous as individual 2-D features
                    pass
            importance_df = pd.DataFrame(
                {
                    "Features": self.config.categorical_cols + self.config.continuous_cols,
                    "importance": imp,
                }
            )
            return importance_df
        else:
            raise ValueError("Feature Importance unavailable for this model.")

    def _read_parse_config(self, config, cls):
        if isinstance(config, str):
            if os.path.exists(config):
                _config = OmegaConf.load(config)
                # if cls == ModelConfig:
                #     cls = getattr_nested(_config._module_src, _config._config_name)
                config = cls(
                    **{
                        k: v
                        for k, v in _config.items()
                        if (k in cls.__dataclass_fields__.keys()) and (cls.__dataclass_fields__[k].init)
                    }
                )
            else:
                raise ValueError(f"{config} is not a valid path")
        # config = OmegaConf.structured(config)
        return config

    def _check_and_verify(self):
        assert hasattr(self, "backbone"), "Model has no attribute called `backbone`"
        assert hasattr(self.backbone, "output_dim"), "Backbone needs to have attribute `output_dim`"
        assert hasattr(self, "head"), "Model has no attribute called `head`"
        
    def _setup_loss(self):
        if self.custom_loss is None:
            try:
                self.loss = getattr(nn, self.config.loss)()
            except AttributeError as e:
                logger.error(f"{self.config.loss} is not a valid loss defined in the torch.nn module")
                raise e
        else:
            self.loss = self.custom_loss

    def _setup_metrics(self):
        if self.custom_metrics is None:
            self.metrics = []
            task_module = torchmetrics.functional
            for metric in self.config.metrics:
                try:
                    self.metrics.append(getattr(task_module, metric))
                except AttributeError as e:
                    logger.error(
                        f"{metric} is not a valid functional metric defined in the torchmetrics.functional module"
                    )
                    raise e
        else:
            self.metrics = self.custom_metrics

    def _get_head_from_config(self):
        _head_callable = LinearHead
        return _head_callable(
            in_units=self.backbone.output_dim,
            output_dim=self.config.output_dim,
            config=_head_callable._config_template(**self.config.head_config),
        )  # output_dim auto-calculated from other configs

    def load_best_model(self) -> None:
        """Loads the best model after training is done."""
        if self.trainer.checkpoint_callback is not None:
            logger.info("Loading the best model")
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            if ckpt_path != "":
                logger.debug(f"Model Checkpoint: {ckpt_path}")
                ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
                self.load_state_dict(ckpt["state_dict"])
            else:
                logger.warning("No best model available to load. Did you run it more than 1 epoch?...")
        else:
            logger.warning("No best model available to load. Checkpoint Callback needs to be enabled for this to work")

    def _prepare_trainer(self, callbacks: List, max_epochs: int = None, min_epochs: int = None) -> pl.Trainer:
        """Prepares the Trainer object
        Args:
            callbacks (List): A list of callbacks to be used
            max_epochs (int, optional): Maximum number of epochs to train for. Defaults to None.
            min_epochs (int, optional): Minimum number of epochs to train for. Defaults to None.

        Returns:
            pl.Trainer: A PyTorch Lightning Trainer object
        """
        logger.info("Preparing the Trainer")
        if max_epochs is not None:
            self.config.max_epochs = max_epochs
        if min_epochs is not None:
            self.config.min_epochs = min_epochs
        # Getting Trainer Arguments from the init signature
        trainer_sig = inspect.signature(pl.Trainer.__init__)
        trainer_args = [p for p in trainer_sig.parameters.keys() if p != "self"]
        trainer_args_config = {k: v for k, v in self.config.items() if k in trainer_args}
        trainer_args_config["enable_checkpointing"] = self.config.enable_checkpointing
        # turn off progress bar if progress_bar=='none'
        trainer_args_config["enable_progress_bar"] = self.config.progress_bar != "none"
        # Adding trainer_kwargs from config to trainer_args
        trainer_args_config.update(self.config.trainer_kwargs)
        return pl.Trainer(
            logger=self.logger,
            callbacks=callbacks,
            **trainer_args_config,
        )
    
    def configure_optimizers(self):
        if self.custom_optimizer is None:
            # Loading from the config
            try:
                self._optimizer = getattr(torch.optim, self.config.optimizer)
                opt = self._optimizer(
                    self.parameters(),
                    lr=self.config.learning_rate,
                    **self.config.optimizer_params,
                )
            except AttributeError as e:
                logger.error(f"{self.config.optimizer} is not a valid optimizer defined in the torch.optim module")
                raise e
        else:
            # Loading from custom fit arguments
            self._optimizer = self.custom_optimizer

            opt = self._optimizer(
                self.parameters(),
                lr=self.config.learning_rate,
                **self.custom_optimizer_params,
            )
        if self.config.lr_scheduler is not None:
            try:
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
            except AttributeError as e:
                logger.error(
                    f"{self.config.lr_scheduler} is not a valid learning rate sheduler defined"
                    f" in the torch.optim.lr_scheduler module"
                )
                raise e
            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(opt, **self.config.lr_scheduler_params),
                }
            return {
                "optimizer": opt,
                "lr_scheduler": self._lr_scheduler(opt, **self.config.lr_scheduler_params),
                "monitor": self.config.lr_scheduler_monitor_metric,
            }
        else:
            return opt


    def _prepare_callbacks(self, callbacks=None) -> List:
        """Prepares the necesary callbacks to the Trainer based on the configuration.

        Returns:
            List: A list of callbacks
        """
        callbacks = [] if callbacks is None else callbacks
        if self.config.early_stopping is not None:
            early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
                monitor=self.config.early_stopping,
                min_delta=self.config.early_stopping_min_delta,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode,
                **self.config.early_stopping_kwargs,
            )
            callbacks.append(early_stop_callback)
        if self.config.checkpoints:
            ckpt_name = f"NewModel-0001"
            ckpt_name = ckpt_name.replace(" ", "_") + "_{epoch}-{valid_loss:.2f}"
            model_checkpoint = pl.callbacks.ModelCheckpoint(
                monitor=self.config.checkpoints,
                dirpath=self.config.checkpoints_path,
                filename=ckpt_name,
                save_top_k=self.config.checkpoints_save_top_k,
                mode=self.config.checkpoints_mode,
                every_n_epochs=self.config.checkpoints_every_n_epochs,
                **self.config.checkpoints_kwargs,
            )
            callbacks.append(model_checkpoint)
            self.config.enable_checkpointing = True
        else:
            self.config.enable_checkpointing = False
        if self.config.progress_bar == "rich" and self.config.trainer_kwargs.get("enable_progress_bar", True):
            callbacks.append(RichProgressBar())
        logger.debug(f"Callbacks used: {callbacks}")
        return callbacks

    def prepare_dataloader(
        self,
        train: pd.DataFrame,
        validation: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        seed: Optional[int] = 42,
    ) -> TabularDatamodule:

        if test is not None:
            warnings.warn(
                "Providing test data in `fit` is deprecated and will be removed in next major release."
                " Plese use `evaluate` for evaluating on test data"
            )
        logger.info("Preparing the DataLoaders")

        datamodule = TabularDatamodule(
            train=train,
            validation=validation,
            config=self.config,
            test=test,
            target_transform=target_transform,
            train_sampler=train_sampler,
            seed=seed,
        )
        #This returns none
        datamodule.prepare_data()
        #This is doing the job of train_test_split
        datamodule.setup("fit")
        return datamodule

    def _prepare_for_training(self, datamodule, callbacks=None, max_epochs=None, min_epochs=None):
        self.callbacks = self._prepare_callbacks(callbacks)
        self.trainer = self._prepare_trainer(self.callbacks, max_epochs, min_epochs)
        self.datamodule = datamodule

    def _train(
        self,
        datamodule: Optional[TabularDatamodule] = None,
        callbacks: Optional[List[pl.Callback]] = None,
        max_epochs: int = None,
        min_epochs: int = None,
    ) -> pl.Trainer:

        self._prepare_for_training(datamodule, callbacks, max_epochs, min_epochs)
        train_loader, val_loader = (
            #Train and val are obtained in the setup() method from TabularDatamodule
            self.datamodule.train_dataloader(),
            self.datamodule.val_dataloader(),
        )
        #from pl.LightningModule
        self.train()
        #
        if self.config.auto_lr_find and (not self.config.fast_dev_run):
            logger.info("Auto LR Find Started")
            result = self.trainer.tune(self, train_loader, val_loader)
            logger.info(
                f"Suggested LR: {result['lr_find'].suggestion()}."
                f" For plot and detailed analysis, use `find_learning_rate` method."
            )

        #from pl.LightningModule
        self.train()
        #
        logger.info("Training Started")
        self.trainer.fit(self, train_loader, val_loader)
        logger.info("Training the model completed")
        if self.config.load_best:
            self.load_best_model()
        return self.trainer

    def _config_metrics(self,config):
        # Updating default metrics in config
        if self.config.task == "classification":
            # Adding metric_params to config for classification task
            for i, mp in enumerate(self.config.metrics_params):
                # For classification task, output_dim == number of classses
                self.config.metrics_params[i]["task"] = mp.get("task", "multiclass")
                self.config.metrics_params[i]["num_classes"] = mp.get("num_classes", config.output_dim)

                if self.config.metrics[i] in (
                    "accuracy",
                    "precision",
                    "recall",
                    "precision_recall",
                    "specificity",
                    "f1_score",
                    "fbeta_score",
                ):
                    self.config.metrics_params[i]["top_k"] = mp.get("top_k", 1)
        
    def fit(
        self,
        train: Optional[pd.DataFrame],
        validation: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
        # loss: Optional[torch.nn.Module] = None,
        # metrics: Optional[List[Callable]] = None,
        # metrics_prob_inputs: Optional[List[bool]] = None,
        # optimizer: Optional[torch.optim.Optimizer] = None,
        # optimizer_params: Dict = {},
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        seed: Optional[int] = 42,
        callbacks: Optional[List[pl.Callback]] = None,
        datamodule: Optional[TabularDatamodule] = None,
    ) -> pl.Trainer:
        seed = seed if seed is not None else self.config.seed
        seed_everything(seed)
        if datamodule is None:
            datamodule = self.prepare_dataloader(train, validation, test, train_sampler, target_transform, seed)
        else:
            if train is not None:
                warnings.warn(
                    "train data is provided but datamodule is provided."
                    " Ignoring the train data and using the datamodule"
                )
            if test is not None:
                warnings.warn(
                    "Providing test data in `fit` is deprecated and will be removed in next major release."
                    " Plese use `evaluate` for evaluating on test data"
                )    
        
        logger.info(f"Preparing the Model")
        inferred_config = self._read_parse_config(datamodule.update_config(self.config), InferredConfig)
        self.config.categorical_dim = inferred_config.categorical_dim
        self.config.continuous_dim = inferred_config.continuous_dim
        self.config.output_dim = inferred_config.output_dim
        self.config.categorical_cardinality = inferred_config.categorical_cardinality
        self.config.embedding_dims = inferred_config.embedding_dims
        self.config.embedded_cat_dim = inferred_config.embedded_cat_dim

        #Setting configurations and saving parameters
        self._setup_loss()
        self._config_metrics(self.config)
        self._setup_metrics()
        #Build the gated neural network structure
        self._build_network()
        self._check_and_verify()
        ########################

        return self._train(datamodule, callbacks, max_epochs, min_epochs)

    def evaluate(
        temp_data,
        model
    ):
        y = temp_data.loc[:,'Prediccion']
        #X=temp_data.drop('Prediccion', axis=1)
        X=temp_data

        # #Initialize the KFold object
        kf = KFold(n_splits=10, shuffle=True, random_state=24)

        # #Accumulate predictions
        all_y_true = []
        all_y_pred = []

        # #Iterate over the folds and train/test the model
        for train_index, val_index in kf.split(temp_data):

            # Get the training and validation data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y.iloc[train_index], y.iloc[val_index]

            #Train model on X_train
            model.fit(train=X_train)

            #Make predictions on X_test
            y_pred = model.predict(X_test)
            y_pred = y_pred.iloc[:,-1]

            #Accumulate values
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        #Evaluate model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
    
    def compute_backbone(self, x: Dict) -> torch.Tensor:
        # Returns output
        x = self.backbone(x)
        return x

    def embed_input(self, x: Dict) -> torch.Tensor:
        return self.embedding_layer(x)

    def pack_output(self, y_hat: torch.Tensor, backbone_features: torch.tensor) -> Dict[str, Any]:
        """Packs the output of the model.

        Args:
            y_hat (torch.Tensor): The output of the model

            backbone_features (torch.tensor): The backbone features

        Returns:
            The packed output of the model
        """
        # if self.head is the Identity function it means that we cannot extract backbone features,
        # because the model cannot be divide in backbone and head (i.e. TabNet)
        if type(self.head) == nn.Identity:
            return {"logits": y_hat}
        return {"logits": y_hat, "backbone_features": backbone_features}

    def compute_head(self, backbone_features: Tensor) -> Dict[str, Any]:
        """Computes the head of the model.

        Args:
            backbone_features (Tensor): The backbone features

        Returns:
            The output of the model
        """
        y_hat = self.head(backbone_features)
        return self.pack_output(y_hat, backbone_features)

    def forward(self, x: Dict) -> Dict[str, Any]:
        """The forward pass of the model.

        Args:
            x (Dict): The input of the model with 'continuous' and 'categorical' keys
        """
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def _predict(self, x: Dict, ret_model_output: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Predicts the output of the model.

        Args:
            x (Dict): The input of the model with 'continuous' and 'categorical' keys

            ret_model_output (bool): If True, the method returns the output of the model

        Returns:
            The output of the model
        """

        ret_value = self.forward(x)
        if ret_model_output:
            return ret_value.get("logits"), ret_value
        return ret_value.get("logits")

    def predict(
        self,
        test: pd.DataFrame,
        ret_logits=False,
        include_input_features: bool = True,
    ) -> pd.DataFrame:
        
        #from pl.LightningModule
        self.eval()

        #Function that prepares and loads the new data. From PyTorchTabular's TabularDataModule class
        inference_dataloader = self.datamodule.prepare_inference_dataloader(test)
        point_predictions = []
        logits_predictions = defaultdict(list)

        #Iterate through the batches
        for batch in track(inference_dataloader, description="Generating Predictions..."):
            for k, v in batch.items():
                if isinstance(v, list) and (len(v) == 0):
                    # Skipping empty list
                    continue
                batch[k] = v.to(self.device)

            #Prediction. y_hat viene del head 
            y_hat, ret_value = self._predict(batch, ret_model_output=True)

            #ret_logits is False by default
            #logit function takes a probability and produces a real number
            #logits are real numbers (0 or 1)
            if ret_logits:
                for k, v in ret_value.items():
                    # if k == "backbone_features":
                    #     continue
                    logits_predictions[k].append(v.detach().cpu())

            point_predictions.append(y_hat.detach().cpu())

        point_predictions = torch.cat(point_predictions, dim=0)
        if point_predictions.ndim == 1:
            point_predictions = point_predictions.unsqueeze(-1)

        if include_input_features:
            pred_df = test.copy()
        else:
            pred_df = pd.DataFrame(index=test.index)

        if self.config.task == "classification":
            #From PyTorch activation functions:
            #nn.Softmax applies the Softmax function to an n-dimensional input Tensor
            #rescaling them so that the elements of the n-dimensional output Tensor
            #lie in the range [0,1] and sum to 1.
            point_predictions = nn.Softmax(dim=-1)(point_predictions).numpy()
            for i, class_ in enumerate(self.datamodule.label_encoder.classes_):
                pred_df[f"{class_}_probability"] = point_predictions[:, i]
            pred_df["prediction"] = self.datamodule.label_encoder.inverse_transform(
                np.argmax(point_predictions, axis=1)
            )

        #ret_logits is False by default
        if ret_logits:
            for k, v in logits_predictions.items():
                v = torch.cat(v, dim=0).numpy()
                if v.ndim == 1:
                    v = v.reshape(-1, 1)
                for i in range(v.shape[-1]):
                    if v.shape[-1] > 1:
                        pred_df[f"{k}_{i}"] = v[:, i]
                    else:
                        pred_df[f"{k}"] = v[:, i]
        return pred_df
    
    def calculate_loss(self, output: Dict, y: torch.Tensor, tag: str) -> torch.Tensor:
        """Calculates the loss for the model.

        Args:
            output (Dict): The output dictionary from the model
            y (torch.Tensor): The target tensor
            tag (str): The tag to use for logging

        Returns:
            torch.Tensor: The loss value
        """
        y_hat = output["logits"]
        reg_terms = [k for k, v in output.items() if "regularization" in k]
        reg_loss = 0
        for t in reg_terms:
            # Log only if non-zero
            if output[t] != 0:
                reg_loss += output[t]
                self.log(
                    f"{tag}_{t}_loss",
                    output[t],
                    on_epoch=True,
                    on_step=False,
                    logger=True,
                    prog_bar=False,
                )

        computed_loss = self.loss(y_hat.squeeze(), y.squeeze()) + reg_loss

        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag in ["valid", "test"]),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        return computed_loss
    

    def calculate_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, tag: str) -> List[torch.Tensor]:
        """Calculates the metrics for the model.

        Args:
            y (torch.Tensor): The target tensor

            y_hat (torch.Tensor): The predicted tensor

            tag (str): The tag to use for logging

        Returns:
            List[torch.Tensor]: The list of metric values
        """
        metrics = []
        for metric, metric_str, prob_inp, metric_params in zip(
            self.metrics,
            self.config.metrics,
            self.config.metrics_prob_input,
            self.config.metrics_params,
        ):
            
            y_hat = nn.Softmax(dim=-1)(y_hat.squeeze())
            if prob_inp:
                avg_metric = metric(y_hat, y.squeeze(), **metric_params)
            else:
                avg_metric = metric(torch.argmax(y_hat, dim=-1), y.squeeze(), **metric_params)
            metrics.append(avg_metric)
            self.log(
                f"{tag}_{metric_str}",
                avg_metric,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True,
            )
        return metrics

    def forward_pass(self, batch):
        return self(batch), None
    
    def training_step(self, batch, batch_idx):
        output, y = self.forward_pass(batch)
        # y is not None for SSL task.Rest of the tasks target is
        # fetched from the batch
        y = batch["target"] if y is None else y
        y_hat = output["logits"]
        loss = self.calculate_loss(output, y, tag="train")
        self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output, y = self.forward_pass(batch)
            # y is not None for SSL task.Rest of the tasks target is
            # fetched from the batch
            y = batch["target"] if y is None else y
            y_hat = output["logits"]
            self.calculate_loss(output, y, tag="valid")
            self.calculate_metrics(y, y_hat, tag="valid")
        return y_hat, y

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output, y = self.forward_pass(batch)
            # y is not None for SSL task.Rest of the tasks target is
            # fetched from the batch
            y = batch["target"] if y is None else y
            y_hat = output["logits"]
            self.calculate_loss(output, y, tag="test")
            self.calculate_metrics(y, y_hat, tag="test")
        return y_hat, y
    
    def save_datamodule(self, dir: str) -> None:
        """Saves the datamodule in the specified directory.

        Args:
            dir (str): The path to the directory to save the datamodule
        """
        joblib.dump(self.datamodule, os.path.join(dir, "datamodule.sav"))
    
    def save_config(self, dir: str) -> None:
        # """Saves the config in the specified directory."""
        # with open(os.path.join(dir, "config.yml"), "w") as fp:
        #     self.config
        """Saves the config in the specified directory."""
        with open(os.path.join(dir, "config.yml"), "w") as fp:
            OmegaConf.save(self.config, fp, resolve=True)


    def save_model(self, dir: str) -> None:
        """Saves the model and checkpoints in the specified directory.

        Args:
            dir (str): The path to the directory to save the model
        """
        if os.path.exists(dir) and (os.listdir(dir)):
            logger.warning("Directory is not empty. Overwriting the contents.")
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        os.makedirs(dir, exist_ok=True)
        self.save_config(dir)
        self.save_datamodule(dir)
        if hasattr(self, "callbacks"):
            joblib.dump(self.callbacks, os.path.join(dir, "callbacks.sav"))
        self.trainer.save_checkpoint(os.path.join(dir, "model.ckpt"))

    def load_model(cls, dir: str, map_location=None, strict=True):
        """Loads a saved model from the directory.

        Args:
            dir (str): The directory where the model wa saved, along with the checkpoints
            map_location (Union[Dict[str, str], str, device, int, Callable, None]) : If your checkpoint
                saved a GPU model and you now load on CPUs or a different number of GPUs, use this to map
                to the new setup. The behaviour is the same as in torch.load()
            strict (bool) : Whether to strictly enforce that the keys in checkpoint_path match the keys
                returned by this module's state dict. Default: True.

        Returns:
            TabularModel (TabularModel): The saved TabularModel
        """
        config = OmegaConf.load(os.path.join(dir, "config.yml"))
        #
        print(config)
        #
        datamodule = joblib.load(os.path.join(dir, "datamodule.sav"))
        logger = None
        if os.path.exists(os.path.join(dir, "callbacks.sav")):
            callbacks = joblib.load(os.path.join(dir, "callbacks.sav"))
            # Excluding Gradient Accumulation Scheduler Callback as we are creating
            # a new one in trainer
            callbacks = [c for c in callbacks if not isinstance(c, GradientAccumulationScheduler)]
        else:
            callbacks = []

        model_args = {
            "config": config,
        }

        model_callable = PredictionModel

        # Initializing with default metrics, losses, and optimizers. Will revert once initialized
        model = model_callable.load_from_checkpoint(
            checkpoint_path=os.path.join(dir, "model.ckpt"),
            map_location=map_location,
            strict=strict,
            **model_args,
        )

        new_model = cls(config=config)
        new_model._setup_metrics()
        # model._setup_loss()
        # new_model.custom_model = custom_model
        new_model.datamodule = datamodule
        new_model.callbacks = callbacks
        new_model.trainer = new_model._prepare_trainer(callbacks=callbacks)
        #Necesary for the trainer init
        new_model.trainer.model = model
        new_model.logger = logger
        return new_model  