import unittest

from models.provider_config import OFFICIAL_BASE_URLS, resolve_provider_connection


class ProviderRoutingTests(unittest.TestCase):
    def test_official_provider_is_default(self):
        key, base_url = resolve_provider_connection(
            "openai",
            environ={"OPENAI_API_KEY": "oa", "APIYI_API_KEY": "px"},
        )

        self.assertEqual(key, "oa")
        self.assertEqual(base_url, OFFICIAL_BASE_URLS["openai"])

    def test_proxy_requires_explicit_url_and_uses_only_proxy_key(self):
        env = {
            "OPENAI_API_KEY": "oa",
            "APIYI_API_KEY": "px",
            "APIYI_BASE_URL": "https://proxy.example/v1",
        }

        key, base_url = resolve_provider_connection("openai", environ=env)

        self.assertEqual(key, "px")
        self.assertEqual(base_url, "https://proxy.example/v1")

    def test_apiyi_hostname_never_receives_official_key(self):
        key, base_url = resolve_provider_connection(
            "google",
            explicit_base_url="https://api.apiyi.com/v1",
            environ={"GOOGLE_API_KEY": "gg"},
        )

        self.assertEqual(key, "")
        self.assertEqual(base_url, "https://api.apiyi.com/v1")

    def test_each_provider_uses_its_own_official_key(self):
        env = {
            "OPENAI_API_KEY": "oa",
            "GOOGLE_API_KEY": "gg",
            "DEEPSEEK_API_KEY": "ds",
        }

        for provider, expected_key in (("openai", "oa"), ("google", "gg"), ("deepseek", "ds")):
            with self.subTest(provider=provider):
                key, base_url = resolve_provider_connection(provider, environ=env)
                self.assertEqual(key, expected_key)
                self.assertEqual(base_url, OFFICIAL_BASE_URLS[provider])

    def test_custom_provider_base_is_an_explicit_choice(self):
        key, base_url = resolve_provider_connection(
            "deepseek",
            environ={
                "DEEPSEEK_API_KEY": "ds",
                "DEEPSEEK_API_BASE": "https://gateway.example/v1/",
            },
        )

        self.assertEqual(key, "ds")
        self.assertEqual(base_url, "https://gateway.example/v1")


if __name__ == "__main__":
    unittest.main()
