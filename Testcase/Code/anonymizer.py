# import re
# import logging
# import sys

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [ANONYMIZER] - %(message)s'
# )

# # --- [ Task 2.2: Anonymization Strategy ] ---
# # This is the "blocklist" of rules, defined as a list of tuples.
# # Each tuple contains: (regex_pattern, replacement_tag)
# #
# # We use regex to find common "secret" patterns.
# # (?i) makes a pattern case-insensitive.
# ANONYMIZATION_RULES = [
#     # 1. Payment Gateway API Keys (e.g., Stripe: sk_live_..., pk_test_...)
#     (r'\b(sk|pk)_(test|live)_[a-zA-Z0-9]{20,}\b', '[GATEWAY_API_KEY]'),

#     # 2. Database Connection Strings (e.g., postgres://user:pass@host:port/db)
#     (r'\b(postgres|mysql|mongodb|sqlite)(s?):\/\/[^:]+:[^@]+@[^\/]+\b', '[DATABASE_CONNECTION_STRING]'),

#     # 3. AWS Access Keys (e.g., AKIA...)
#     (r'\b(AKIA[A-Z0-9]{16})\b', '[AWS_ACCESS_KEY]'),
    
#     # 4. Generic High-Entropy Hashes (e.g., 64-char SHA-256)
#     # This finds long hexadecimal strings, often used for secrets.
#     (r'\b[A-Fa-f0-9]{40,}\b', '[GENERIC_SECRET_HASH]'),

#     # 5. Common Currency Codes (as requested in your document)
#     (r'\b(USD|EUR|GBP|INR|JPY|CAD|AUD)\b', '[CURRENCY]'),

#     # 6. Proprietary Keywords (e.g., "INTERNAL_PROJECT_NAME")
#     # For your shopping codebase, you might add 'SHOPPING_APP'
#     (r'\b(ACME_CORP|PROJECT_PHOENIX|SHOPPING_APP)\b', '[PROPRIETARY_NAME]'),
    
#     # 7. Common variable names for secrets (replaces the value, not the key)
#     # Looks for: API_KEY = "value"
#     (r'(?i)(api_key|secret|password)\s*=\s*["\'](.*?)["\']', r'\1 = "[SECRET_VALUE]"')
# ]
# # --- [ End of Task 2.2 ] ---


# def anonymize_code_chunk(text: str) -> str:
#     """
#     Applies all defined ANONYMIZATION_RULES to a given text chunk.

#     Args:
#         text: A string containing the raw code chunk.

#     Returns:
#         A new string with all found patterns replaced by their tags.
#     """
#     if not isinstance(text, str):
#         logging.warning(f"Anonymizer received non-string input. Type: {type(text)}. Skipping.")
#         return str(text) # Return string-ified version just in case

#     anonymized_text = text
#     for pattern, replacement in ANONYMIZATION_RULES:
#         try:
#             anonymized_text = re.sub(pattern, replacement, anonymized_text)
#         except re.error as e:
#             # This helps debug a bad regex pattern
#             logging.error(f"Regex error in pattern: {pattern}. Error: {e}")
#         except Exception as e:
#             logging.error(f"Unexpected error during anonymization: {e}")
            
#     return anonymized_text

# # Note: The test_anonymizer() function and the
# # if __name__ == "__main__": block have been removed
# # as requested, to make this a clean importable module.





import re
import logging

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [ANONYMIZER] - %(message)s'
)

# --- [ Task 2.2: Anonymization Strategy ] ---
#
# This is the "blocklist" of rules. We define a regular expression
# to *find* a secret, and a "tag" to replace it with.
#
# We are specifically adding rules for:
# 1. API key prefixes (like 'sk_test_')
# 2. Config variables (like config.STRIPE_API_KEY)
# 3. Database connection strings
# 4. Common currency codes
#
ANONYMIZATION_RULES = [
    # Rule 1: Find API Key Prefixes (e.g., 'sk_test_', "pk_live_")
    # This matches ' or " followed by sk_test_, pk_test_, sk_live_, etc.
    (re.compile(r'["\']([sp]k_(test|live)_)["\']'), r'"[GATEWAY_API_KEY_PREFIX]"'),

    # Rule 2: Find full API Keys in strings (e.g., "sk_live_...1234")
    (re.compile(r'["\']([sp]k_(test|live)_[a-zA-Z0-9_]+)["\']'), r'"[GATEWAY_API_KEY]"'),

    # Rule 3: Find config variables (e.g., config.STRIPE_API_KEY)
    (re.compile(r'config\.[A-Z_]+_API_KEY'), '[GATEWAY_API_KEY]'),
    (re.compile(r'config\.[A-Z_]+_URI'), '[DATABASE_CONNECTION_STRING]'),

    # Rule 4: Find database connection strings
    (re.compile(r'["\'](postgres|mysql|mongodb)([^"\']+)["\']'), r'"[DATABASE_CONNECTION_STRING]"'),
    
    # Rule 5: Find common currency codes (as whole words)
    (re.compile(r'\b(USD|EUR|GBP)\b'), r'[CURRENCY]')
]
# --- End Strategy ---


def anonymize_code_chunk(code_chunk: str) -> str:
    """
    Iterates through all ANONYMIZATION_RULES and applies them
    to a single chunk of code or text.
    
    This is the core function for Task 2.2.
    """
    if not code_chunk:
        return ""
        
    anonymized_chunk = code_chunk
    
    for rule, replacement_tag in ANONYMIZATION_RULES:
        try:
            # Find all matches and replace them with the tag
            anonymized_chunk = rule.sub(replacement_tag, anonymized_chunk)
        except re.error as e:
            # This helps debug a bad regex rule
            logging.warning(f"Regex error in anonymizer: {e}")
            
    return anonymized_chunk


# --- Self-Test Block (Optional) ---
# You can run this file directly (`python anonymizer.py`)
# to test if the rules above are working correctly.
def test_anonymizer():
    """
    Tests the ANONYMIZATION_RULES against a sample of "dirty" code.
    """
    logging.info("--- [ Testing Anonymization Rules ] ---")
    
    # This sample includes the *exact* secrets from your file
    dirty_code = """
    # This has config.STRIPE_API_KEY
    def __init__(self):
        self.api_key = config.STRIPE_API_KEY
        print(f"PaymentService initialized for Stripe.")

    # This has 'sk_test_'
    def process_payment(self, amount: float, currency: str) -> bool:
        if self.api_key.startswith('sk_test_'):
            return True
            
    # This has a currency and a full key
    def new_charge(self):
        charge = create_charge(amount=100, currency="USD", key="sk_live_123abc789")
        db = "postgres://user:pass@host.com/db"
    """
    
    clean_code = anonymize_code_chunk(dirty_code)
    
    print("\n--- [ ORIGINAL 'Dirty' Code ] ---")
    print(dirty_code)
    print("\n--- [ ANONYMIZED 'Clean' Code ] ---")
    print(clean_code)
    print("\n--- [ Test Results ] ---")
    
    if "[GATEWAY_API_KEY]" in clean_code and \
       "[GATEWAY_API_KEY_PREFIX]" in clean_code and \
       "[CURRENCY]" in clean_code and \
       "[DATABASE_CONNECTION_STRING]" in clean_code:
        logging.info("PASS: All test secrets were successfully anonymized.")
    else:
        logging.error("FAIL: One or more secrets were NOT anonymized.")
    logging.info("---------------------------------------")

if __name__ == "__main__":
    # This block runs *only* if you execute `python anonymizer.py`
    test_anonymizer()

