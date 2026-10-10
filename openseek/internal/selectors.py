from dataclasses import dataclass, field


@dataclass
class LoginSelectors:
    email_input: str = 'input[type="text"]'
    password_input: str = 'input[type="password"]'
    login_button: str = 'div.ds-button--primary.ds-button--xl'
    cookie_dismiss_button: str = '.cookie_banner-accept-essential-button'
    verification_text: str = 'Verify your account'


@dataclass
class InteractionSelectors:
    textbox: str = 'textarea._27c9245'
    send_button: str = 'div._52c986b'
    file_input: str = 'input[type="file"]'


@dataclass
class DeepSeekSelectors:
    login: LoginSelectors = field(default_factory=LoginSelectors)
    interactions: InteractionSelectors = field(default_factory=InteractionSelectors)
