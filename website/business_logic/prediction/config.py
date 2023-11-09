from typing import Dict, List, Optional, Any, Optional,ItemsView, Iterable
from dataclasses import dataclass, field, asdict
from pytorch_tabular.utils import get_logger
from pytorch_tabular.models.common import heads


logger = get_logger(__name__)

def _validate_choices(cls):
    for key in cls.__dataclass_fields__.keys():
        atr = cls.__dataclass_fields__[key]
        if atr.init and "choices" in atr.metadata.keys():
            if getattr(cls, key) not in atr.metadata.get("choices"):
                raise ValueError(
                    f"{getattr(cls, key)} is not a valid choice for {key}."
                    f" Please choose from on of the following: {atr.metadata['choices']}"
                )

@dataclass
class PredictionModelConfig:

    categorical_dim: int = field(
        metadata={"help": "The number of categorical features"},
    )
    continuous_dim: int = field(
        metadata={"help": "The number of continuous features"},
    )
    continuous_cols: List = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_cols: List = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields to treat differently. Defaults to []"},
    )
    task: str = field(
        default="classification",
        metadata={
            "help": "Specify whether the problem is regression or classification."
            " `backbone` is a task which considers the model as a backbone to generate features."
            " Mostly used internally for SSL and related tasks.",
            "choices": ["regression", "classification", "backbone"],
        }
    )
    head: Optional[str] = field(
        default="LinearHead",
        metadata={
            "help": "The head to be used for the model. Should be one of the heads defined"
            " in `pytorch_tabular.models.common.heads`. Defaults to  LinearHead",
            "choices": [None, "LinearHead", "MixtureDensityHead"],
        },
    )
    head_config: Optional[Dict] = field(
        default_factory=lambda: {"layers": "256-64"},
        metadata={
            "help": "The config as a dict which defines the head."
            " If left empty, will be initialized as default linear head."
        },
    )
    layers: str = field(
        default="64-32-16",
        metadata={
            "help": "Hyphen-separated number of layers and units in the classification head. eg. 32-64-32."
            " Defaults to 64-32-16"
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch"
            " like ReLU, TanH, LeakyReLU, etc."
            " https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity."
            " Defaults to ReLU"
        },
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults to False"},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers. Defaults to `kaiming`",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "probability of an classification element to be zeroed."
            " This is added to each linear layer. Defaults to 0.0"
        },
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim). If left empty, will infer using the cardinality of the "
            "categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout to be applied to the Categorical Embedding. Defaults to 0.0"},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={"help": "If True, we will normalize the continuous layer by passing it through a BatchNorm layer."},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "The learning rate of the model. Defaults to 1e-3."},
    )
    loss: Optional[str] = field(
        default="CrossEntropyLoss",
        metadata={
            "help": "The loss function to be applied. By Default it is MSELoss for regression "
            "and CrossEntropyLoss for classification. Unless you are sure what you are doing, "
            "leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification"
        },
    )
    metrics: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "the list of metrics you need to track during training. The metrics should be one "
            "of the functional metrics implemented in ``torchmetrics``. To use your own metric, please "
            "use the `metric` param in the `fit` method By default, it is accuracy if classification "
            "and mean_squared_error for regression"
        },
    )
    metrics_prob_input: Optional[List[bool]] = field(
        default=None,
        metadata={
            "help": "Is a mandatory parameter for classification metrics defined in the config. This defines "
            "whether the input to the metric function is the probability or the class. Length should be same "
            "as the number of metrics. Defaults to None."
        },
    )
    metrics_params: Optional[List] = field(
        default=None,
        metadata={
            "help": "The parameters to be passed to the metrics function. `task` is forced to be `multiclass`` "
            "because the multiclass version can handle binary as well and for simplicity we are only using "
            "`multiclass`."
        },
    )
    target_range: Optional[List] = field(
        default=None,
        metadata={
            "help": "The range in which we should limit the output variable. "
            "Currently ignored for multi-target regression. Typically used for Regression problems. "
            "If left empty, will not apply any restrictions"
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "The seed for reproducibility. Defaults to 42"},
    )
    gflu_stages: int = field(
        default=6,
        metadata={"help": "Number of layers in the feature abstraction layer. Defaults to 6"},
    )
    gflu_dropout: float = field(
        default=0.0, 
        metadata={"help": "Dropout rate for the feature abstraction layer. Defaults to 0.0"}
    )
    gflu_feature_init_sparsity: float = field(
        default=0.3,
        metadata={
            "help": "Only valid for t-softmax. The perecentge of features to be selected in "
            "each GFLU stage. This is just initialized and during learning it may change"
        },
    )
    learnable_sparsity: bool = field(
        default=True,
        metadata={
            "help": "Only valid for t-softmax. If True, the sparsity parameters will be learned."
            "If False, the sparsity parameters will be fixed to the initial values specified in "
            "`gflu_feature_init_sparsity` and `tree_feature_init_sparsity`"
        },
    )
    target: Optional[List[str]] = field(
        default_factory=lambda: ['Prediccion'],
        metadata={
            "help": "A list of strings with the names of the target column(s)."
            " It is mandatory for all except SSL tasks."
        },
    )
    date_columns: List = field(
        default_factory=list,
        metadata={
            "help": "(Column names, Freq) tuples of the date fields. For eg. a field named"
            " `introduction_date` and with a monthly frequency should have an entry ('intro_date','M'}"
        },
    )
    encode_date_columns: bool = field(
        default=True,
        metadata={"help": "Whether or not to encode the derived variables from date"},
    )
    validation_split: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Percentage of Training rows to keep aside as validation."
            " Used only if Validation Data is not given separately"
        },
    )
    continuous_feature_transform: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether or not to transform the features before modelling. By default it is turned off.",
            "choices": [
                None,
                "yeo-johnson",
                "box-cox",
                "quantile_normal",
                "quantile_uniform",
            ],
        },
    )
    normalize_continuous_features: bool = field(
        default=True,
        metadata={"help": "Flag to normalize the input features(continuous)"},
    )
    quantile_noise: int = field(
        default=0,
        metadata={
            "help": "NOT IMPLEMENTED. If specified fits QuantileTransformer on data with added gaussian noise"
            " with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable."
            " Please not that this transformation does NOT apply gaussian noise to the resulting data,"
            " the noise is only applied for QuantileTransformer"
        },
    )
    num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of workers used for data loading. For windows always set to 0"},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to pin memory for data loading."},
    )
    handle_unknown_categories: bool = field(
        default=True,
        metadata={"help": "Whether or not to handle unknown or new values in categorical columns as unknown"},
    )
    handle_missing_values: bool = field(
        default=True,
        metadata={"help": "Whether or not to handle missing values in categorical columns as unknown"},
    )
    batch_size: int = field(
        default=64, 
        metadata={"help": "Number of samples in each batch of training"}
    )
    data_aware_init_batch_size: int = field(
        default=2000,
        metadata={
            "help": "Number of samples in each batch of training for the data-aware initialization,"
            " when applicable. Defaults to 2000"
        },
    )
    fast_dev_run: bool = field(
        default=False,
        metadata={
            "help": "runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train,"
            " val and test to find any bugs (ie: a sort of unit test)."
        },
    )
    max_epochs: int = field(
        default=10, 
        metadata={"help": "Maximum number of epochs to be run"}
    )
    min_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Force training for at least these many epochs. 1 by default"},
    )
    max_time: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training after this amount of time has passed. Disabled by default (None)"},
    )
    gpus: Optional[int] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: Number of gpus to train on (int). -1 uses all available GPUs."
            " By default uses CPU (None)"
        },
    )
    accelerator: Optional[str] = field(
        default="cpu",
        metadata={
            "help": "The accelerator to use for training. Can be one of 'cpu','gpu','tpu','ipu','auto'."
            " Defaults to 'auto'",
            "choices": ["cpu", "gpu", "tpu", "ipu", "mps", "auto"],
        },
    )
    devices: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of devices to train on (int). -1 uses all available devices."
            " By default uses all available devices (-1)",
        },
    )
    devices_list: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of devices to train on (list). If specified, takes precedence over `devices` argument."
            " Defaults to None",
        },
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={
            "help": "Accumulates grads every k batches or as set up in the dict."
            " Trainer also calls optimizer.step() for the last indivisible step number."
        },
    )
    auto_lr_find: bool = field(
        default=False,
        metadata={
            "help": "Runs a learning rate finder algorithm (see this paper) when calling trainer.tune(),"
            " to find optimal initial learning rate."
        },
    )
    auto_select_gpus: bool = field(
        default=False,
        metadata={
            "help": "If enabled and `devices` is an integer, pick available gpus automatically."
            " This is especially useful when GPUs are configured to be in 'exclusive mode',"
            " such that only one process at a time can access them."
        },
    )
    check_val_every_n_epoch: int = field(
        default=1, 
        metadata={"help": "Check val every n train epochs."}
    )
    gradient_clip_val: float = field(
        default=0.0, 
        metadata={"help": "Gradient clipping value"}
    )
    overfit_batches: float = field(
        default=0.0,
        metadata={
            "help": "Uses this much data of the training set. If nonzero, will use the same training set"
            " for validation and testing. If the training dataloaders have shuffle=True,"
            " Lightning will automatically disable it."
            " Useful for quickly debugging or trying to overfit on purpose."
        },
    )
    deterministic: bool = field(
        default=False,
        metadata={
            "help": "If true enables cudnn.deterministic. Might make your system slower, but ensures reproducibility."
        },
    )
    profiler: Optional[str] = field(
        default=None,
        metadata={
            "help": "To profile individual steps during training and assist in identifying bottlenecks."
            " None, simple or advanced, pytorch",
            "choices": [None, "simple", "advanced", "pytorch"],
        },
    )
    early_stopping: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "The loss/metric that needed to be monitored for early stopping."
            " If None, there will be no early stopping"
        },
    )
    early_stopping_min_delta: float = field(
        default=0.001,
        metadata={"help": "The minimum delta in the loss/metric which qualifies as an improvement in early stopping"},
    )
    early_stopping_mode: str = field(
        default="min",
        metadata={
            "help": "The direction in which the loss/metric should be optimized",
            "choices": ["max", "min"],
        },
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "The number of epochs to wait until there is no further improvements in loss/metric"},
    )
    early_stopping_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {},
        metadata={
            "help": "Additional keyword arguments for the early stopping callback."
            " See the documentation for the PyTorch Lightning EarlyStopping callback for more details."
        },
    )
    checkpoints: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "The loss/metric that needed to be monitored for checkpoints. If None, there will be no checkpoints"
        },
    )
    checkpoints_path: str = field(
        default="saved_models",
        metadata={"help": "The path where the saved models will be"},
    )
    checkpoints_every_n_epochs: int = field(
        default=1,
        metadata={"help": "Number of training steps between checkpoints"},
    )
    checkpoints_name: Optional[str] = field(
        default="tesina_model",
        metadata={
            "help": "The name under which the models will be saved. If left blank,"
            " first it will look for `run_name` in experiment_config and if that is also None"
            " then it will use a generic name like task_version."
        },
    )
    checkpoints_mode: str = field(
        default="min",
        metadata={"help": "The direction in which the loss/metric should be optimized"},
    )
    checkpoints_save_top_k: int = field(
        default=1,
        metadata={"help": "The number of best models to save"},
    )
    checkpoints_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {},
        metadata={
            "help": "Additional keyword arguments for the checkpoints callback. See the documentation"
            " for the PyTorch Lightning ModelCheckpoint callback for more details."
        },
    )
    load_best: bool = field(
        default=True,
        metadata={"help": "Flag to load the best model saved during training"},
    )
    track_grad_norm: int = field(
        default=-1,
        metadata={
            "help": "Track and Log Gradient Norms in the logger. -1 by default means no tracking. "
            "1 for the L1 norm, 2 for L2 norm, etc."
        },
    )
    progress_bar: str = field(
        default="rich",
        metadata={"help": "Progress bar type. Can be one of: `none`, `simple`, `rich`. Defaults to `rich`."},
    )
    precision: int = field(
        default=32,
        metadata={
            "help": "Precision of the model. Can be one of: `32`, `16`, `64`. Defaults to `32`.",
            "choices": [32, 16, 64],
        },
    )
    trainer_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Additional kwargs to be passed to PyTorch Lightning Trainer."},
    )
    optimizer: str = field(
        default="Adam",
        metadata={
            "help": "Any of the standard optimizers from"
            " [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms)."
        },
    )
    optimizer_params: Dict = field(
        default_factory=lambda: {},
        metadata={"help": "The parameters for the optimizer. If left blank, will use default parameters."},
    )
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the LearningRateScheduler to use, if any, from"
            " https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate."
            " If None, will not use any scheduler. Defaults to `None`",
        },
    )
    lr_scheduler_params: Optional[Dict] = field(
        default_factory=lambda: {},
        metadata={"help": "The parameters for the LearningRateScheduler. If left blank, will use default parameters."},
    )
    lr_scheduler_monitor_metric: Optional[str] = field(
        default="valid_loss",
        metadata={"help": "Used with ReduceLROnPlateau, where the plateau is decided based on this metric"},
    )
    output_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The number of output targets"},
    )
    categorical_cardinality: Optional[List[int]] = field(
        default=None,
        metadata={"help": "The number of unique values in categorical features"},
    )
    embedded_cat_dim: int = field(
        init=False,
        metadata={"help": "The number of features or dimensions of the embedded categorical features"},
    )
    enable_checkpointing: bool = field(
        default=True,
    )
    

    def __post_init__(self):
        assert (
            len(self.categorical_cols) + len(self.continuous_cols) + len(self.date_columns) > 0
        ), "There should be at-least one feature defined in categorical, continuous, or date columns"

        assert (
            self.task == "classification"
        ), "The task should not be empty."

        # if self.task == "regression":
        #     self.loss = self.loss or "MSELoss"
        #     self.metrics = self.metrics or ["mean_squared_error"]
        #     self.metrics_params = [{} for _ in self.metrics] if self.metrics_params is None else self.metrics_params
        #     self.metrics_prob_input = [False for _ in self.metrics]  # not used in Regression. just for compatibility
        # el
        if self.task == "classification":
            self.loss = self.loss or "CrossEntropyLoss"
            self.metrics = self.metrics or ["accuracy"]
            self.metrics_params = [{} for _ in self.metrics] if self.metrics_params is None else self.metrics_params
            self.metrics_prob_input = (
                [False for _ in self.metrics] if self.metrics_prob_input is None else self.metrics_prob_input
            )
        # elif self.task == "backbone":
        #     self.loss = None
        #     self.metrics = None
        #     self.metrics_params = None
        #     if self.head is not None:
        #         logger.warning("`head` is not a valid parameter for backbone task. Making `head=None`")
        #         self.head = None
        #         self.head_config = None
        else:
            raise NotImplementedError(
                f"{self.task} is not a valid task. Should be one of "
                f"{self.__dataclass_fields__['task'].metadata['choices']}"
            )
        if self.metrics is not None:
            assert len(self.metrics) == len(self.metrics_params), "metrics and metric_params should have same length"

        if self.task != "backbone":
            assert self.head in dir(heads.blocks), f"{self.head} is not a valid head"
            _head_callable = getattr(heads.blocks, self.head)
            ideal_head_config = _head_callable._config_template
            invalid_keys = set(self.head_config.keys()) - set(ideal_head_config.__dict__.keys())
            assert len(invalid_keys) == 0, f"`head_config` has some invalid keys: {invalid_keys}"

        # For Custom models, setting these values for compatibility
        # if not hasattr(self, "_config_name"):
        #     self._config_name = type(self).__name__
        # if not hasattr(self, "_model_name"):
        #     self._model_name = re.sub("[Cc]onfig", "Model", self._config_name)
        # if not hasattr(self, "_backbone_name"):
        #     self._backbone_name = re.sub("[Cc]onfig", "Backbone", self._config_name)
        # _validate_choices(self)
        # if self.gpus is not None:
        #     warnings.warn(
        #         "The `gpus` argument is deprecated in favor of `accelerator` and will be removed in a future version"
        #         " of PyTorch Tabular. Please use `accelerator='gpu'` instead.",
        #         DeprecationWarning,
        #     )
        #     if self.devices is None:
        #         self.devices = self.gpus
        #     if self.accelerator is None:
        #         self.accelerator = "gpu"
        # else:
        if self.accelerator is None:
            self.accelerator = "cpu"
        # delattr(self, "gpus")
        # if self.devices_list is not None:
        #     warnings.warn("Ignoring devices in favor of devices_list")
        #     self.devices = self.devices_list
        # delattr(self, "devices_list")
        for key in self.early_stopping_kwargs.keys():
            if key in ["min_delta", "mode", "patience"]:
                raise ValueError(
                    f"Cannot override {key} in early_stopping_kwargs."
                    f" Please use the appropriate argument in `TrainerConfig`"
                )
        for key in self.checkpoints_kwargs.keys():
            if key in ["dirpath", "filename", "monitor", "save_top_k", "mode", "every_n_epochs"]:
                raise ValueError(
                    f"Cannot override {key} in checkpoints_kwargs."
                    f" Please use the appropriate argument in `TrainerConfig`"
                )
        if self.embedding_dims is not None:
            assert all(
                (isinstance(t, Iterable) and len(t) == 2) for t in self.embedding_dims
            ), "embedding_dims must be a list of tuples (cardinality, embedding_dim)"
            self.embedded_cat_dim = sum([t[1] for t in self.embedding_dims])
        else:
            self.embedded_cat_dim = 0

        _validate_choices(self)

    def items(self) -> ItemsView[str, Any]:
        return asdict(self).items()