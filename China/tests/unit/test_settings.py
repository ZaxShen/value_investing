"""
Unit tests for the settings configuration module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.settings import configure_environment, get_config, is_tqdm_enabled


class TestConfigureEnvironment:
    """Test environment configuration functionality."""

    @pytest.mark.unit
    def test_configure_environment_enables_tqdm(self):
        """Test that configure_environment enables tqdm by removing TQDM_DISABLE."""
        with patch.dict(os.environ, {"TQDM_DISABLE": "1"}, clear=True):
            # Start with TQDM_DISABLE set
            assert "TQDM_DISABLE" in os.environ
            
            config = configure_environment()
            
            assert config["tqdm_enabled"] == True
            assert "TQDM_DISABLE" not in os.environ

    @pytest.mark.unit
    def test_configure_environment_returns_dict(self):
        """Test that configure_environment returns a configuration dictionary."""
        config = configure_environment()
        
        assert isinstance(config, dict)
        assert "tqdm_enabled" in config

    @pytest.mark.unit
    def test_configure_environment_idempotent(self):
        """Test that calling configure_environment multiple times is safe."""
        config1 = configure_environment()
        config2 = configure_environment()
        
        assert config1 == config2
        assert "TQDM_DISABLE" not in os.environ

    @pytest.mark.unit
    def test_configure_environment_preserves_existing_env_vars(self):
        """Test that configure_environment doesn't clear existing environment variables."""
        test_var = "TEST_PRESERVE_VAR"
        test_value = "preserve_me"
        
        with patch.dict(os.environ, {test_var: test_value, "TQDM_DISABLE": "1"}):
            configure_environment()
            
            # Original var should still be there
            assert os.environ.get(test_var) == test_value
            # TQDM_DISABLE should be removed
            assert "TQDM_DISABLE" not in os.environ


class TestGetConfig:
    """Test configuration retrieval functionality."""

    @pytest.mark.unit
    def test_get_config_returns_copy(self):
        """Test that get_config returns a copy of the configuration."""
        config1 = get_config()
        config2 = get_config()
        
        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2

    @pytest.mark.unit
    def test_get_config_modification_safe(self):
        """Test that modifying returned config doesn't affect internal state."""
        config = get_config()
        original_config = get_config()
        
        # Modify the returned config
        config["test_key"] = "test_value"
        
        # Get config again - should be unchanged
        new_config = get_config()
        assert new_config == original_config
        assert "test_key" not in new_config

    @pytest.mark.unit
    def test_get_config_contains_expected_keys(self):
        """Test that get_config returns expected configuration keys."""
        config = get_config()
        
        assert "tqdm_enabled" in config
        assert isinstance(config["tqdm_enabled"], bool)


class TestIsTqdmEnabled:
    """Test tqdm enabled check functionality."""

    @pytest.mark.unit
    def test_is_tqdm_enabled_returns_bool(self):
        """Test that is_tqdm_enabled returns a boolean value."""
        result = is_tqdm_enabled()
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_is_tqdm_enabled_reflects_config(self):
        """Test that is_tqdm_enabled reflects the actual configuration."""
        config = get_config()
        result = is_tqdm_enabled()
        
        assert result == config.get("tqdm_enabled", True)

    @pytest.mark.unit
    def test_is_tqdm_enabled_default_true_when_missing(self):
        """Test that is_tqdm_enabled returns True when tqdm_enabled key is missing."""
        with patch("src.settings._config", {}):
            result = is_tqdm_enabled()
            assert result == True


class TestSettingsIntegration:
    """Integration tests for settings module."""

    @pytest.mark.integration
    def test_settings_auto_configuration_on_import(self):
        """Test that settings are automatically configured when module is imported."""
        # Since the module auto-configures on import, check that it worked
        assert "TQDM_DISABLE" not in os.environ
        
        config = get_config()
        assert config["tqdm_enabled"] == True

    @pytest.mark.integration
    def test_environment_variable_persistence(self):
        """Test that TQDM_DISABLE removal persists."""
        # The module should have already removed TQDM_DISABLE
        initial_value = os.environ.get("TQDM_DISABLE")
        assert initial_value is None
        
        # Configure again - should maintain the absence
        configure_environment()
        after_config = os.environ.get("TQDM_DISABLE")
        assert after_config is None

    @pytest.mark.integration
    def test_settings_affect_tqdm_behavior(self):
        """Test that TQDM_DISABLE setting actually affects tqdm (if available)."""
        try:
            import tqdm
            # If tqdm is available, check that it respects the setting
            # This is more of a smoke test since tqdm behavior depends on internal implementation
            assert "TQDM_DISABLE" not in os.environ
            assert is_tqdm_enabled() == True
        except ImportError:
            # tqdm not available, skip this specific test
            pytest.skip("tqdm not available for testing")

    @pytest.mark.integration  
    def test_settings_module_as_first_import(self):
        """Test that settings can be safely imported as first module."""
        # This test ensures that the settings module doesn't have problematic dependencies
        # that would prevent it from being imported first
        
        # The module should be already imported and configured
        assert get_config() is not None
        assert is_tqdm_enabled() is not None
        
        # Re-importing should be safe
        import src.settings
        assert src.settings.get_config() is not None


class TestSettingsEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_configure_environment_with_existing_tqdm_disable(self):
        """Test configure_environment when TQDM_DISABLE already exists."""
        with patch.dict(os.environ, {"TQDM_DISABLE": "0"}):
            config = configure_environment()
            
            # Should remove the existing TQDM_DISABLE
            assert config["tqdm_enabled"] == True
            assert "TQDM_DISABLE" not in os.environ

    @pytest.mark.unit
    def test_get_config_with_corrupted_internal_config(self):
        """Test get_config behavior with corrupted internal config."""
        with patch("src.settings._config", None):
            # This would likely cause an AttributeError, which is expected behavior
            with pytest.raises(AttributeError):
                get_config()

    @pytest.mark.unit
    def test_is_tqdm_enabled_with_non_bool_value(self):
        """Test is_tqdm_enabled with non-boolean value in config."""
        with patch("src.settings._config", {"tqdm_enabled": "true"}):
            result = is_tqdm_enabled()
            # Should return the actual value (string in this case)
            assert result == "true"

    @pytest.mark.unit
    def test_configure_environment_robust_error_handling(self):
        """Test configure_environment handles various error conditions."""
        # Test that the function can handle unexpected conditions gracefully
        # This is more of a smoke test since the function is quite simple
        config = configure_environment()
        assert isinstance(config, dict)
        assert "tqdm_enabled" in config