# utils/notifications.py

import aiohttp
import logging

async def send_slack_message(webhook_url: str, message: str):
    """
    Sends a message to a Slack channel using webhook URL.
    """
    if not webhook_url:
        logging.warning("Slack webhook URL not configured.")
        return

    payload = {"text": message}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logging.warning(f"Slack message failed: {response.status}")
                else:
                    logging.info("Slack message sent.")
    except Exception as e:
        logging.error(f"Slack error: {e}")


async def send_email(smtp_config: dict, subject: str, body: str):
    """Placeholder for sending email alerts via SMTP."""
    logging.info("Email alerts not configured. Stub executed.")
