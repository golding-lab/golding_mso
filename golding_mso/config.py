
import json
import logging
from importlib.resources import path
from shutil import copy


logger = logging.getLogger(__name__)

class Config(dict):
    """
    Configuration class for managing golding_mso package settings.
    """
    
    def __init__(self, user_path: str = None):
        """
        Initialize the Config object by loading the configuration from the specified path or the default user config path
        Parameters
        ----------
        user_path : str, optional
            The file path to load the configuration from. If None, uses the default user config path
        """
        self.user_path = user_path
        super().__init__()
        self.update(self.from_user)
    
    def rebase(self, new_user_path: str):
        """
        Change the user configuration path and reload the configuration.

        Parameters
        ----------
        new_user_path : str
            The new file path to load the configuration from.
        """
        self.user_path = new_user_path
        self.reload(config_path=new_user_path)
        
    def reload(self, config: dict = None, config_path: str = None):
        """
        Load a configuration from a specified file path.

        Parameters
        ----------
        config : dict, optional
            A dictionary representing the configuration to load. If config_path is provided, this parameter is ignored.
        config_path : str, optional
            The file path to load the configuration from. If None, uses the default user config path.
        """

        if config is not None and config_path is None:
            loaded_config = config
        else:
            if config_path is None:
                config_path = self.path
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
        
        self.clear()
        self.update(loaded_config)
        logger.info(f"Configuration loaded from {'dict' if config_path is None and config is not None else 'path'}.")
        
    def reload_default(self):
        """
        Reload the default configuration from the package's default configuration file.
        """
        default_config = self.from_default
        self.clear()
        self.update(default_config)
        logger.info("Default configuration reloaded.")
    
    def restore_default_file(self) -> dict:
        """
        Reset the user configuration file to its default.
        """
        
        with path("golding_mso", "golding_mso_config_default.json") as dcp:
            def_config_path = dcp
        try:
            if not self.user_path.is_dir():
                # Create the user config directory if it does not exist
                self.user_path.mkdir(parents=True, exist_ok=True)
            copy(
                def_config_path,
                self.user_path / "golding_mso_config.json",
            )
            copy(
                def_config_path,
                self.user_path / "golding_mso_config.json",
            )
            logger.info("Default configuration files copied to user config directory.")
        except FileNotFoundError:
            logger.exception(
                "Default configuration file 'golding_mso_config_default.json' not found. "
                "You may need to reinstall the golding_mso package or replace the file manually.",
            )

    def save(self, new_config: dict = None):
        """
        Set the configuration of the golding_mso package by saving the current session's config or a provided dictionary.
        
        Parameters
        ----------
        new_config : dict, optional
            A dictionary representing the new configuration to save. If None, saves the current session's config.
        """
        
        if new_config is None:
            new_config=self
            
        config_path = self.user_path / "golding_mso_config.json"
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
        
    @property
    def from_user(self) -> dict:
        """
        Load the user's configuration from the configuration file.

        Returns
        -------
        dict:
            The user's configuration as a dictionary.
        """
        config_path = self.user_path / "golding_mso_config.json"
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            return user_config
        except:
            logger.error(
                "Configuration file 'golding_mso_config.json' not found in user config path. Loading default configuration. Try running reset_config() to create the user's default configuration."
            )
            return self.from_default
        
    @property
    def from_default(self) -> dict:
        """
        Load the default configuration from the package's default configuration file.

        Returns
        -------
        dict:
            The default configuration as a dictionary.
        """
        with path("golding_mso", "golding_mso_config_default.json") as dcp:
            default_config_path = dcp
        with open(default_config_path, "r") as f:
            default_config = json.load(f)
        return default_config



