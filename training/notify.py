"""Slack webhook notifications for training runs."""

from slack_sdk.webhook import WebhookClient


def send(webhook_url, message):
    """POST a message to Slack. Silently no-ops on missing URL or errors."""
    if not webhook_url:
        return
    try:
        client = WebhookClient(webhook_url)
        client.send(text=message)
    except Exception:
        pass


def training_started(webhook_url, run_name, model, phases, gpu_type, sha):
    msg = (
        f"\U0001f680 *{run_name}* started\n"
        f"Model: {model} | Phases: {phases} | GPU: {gpu_type} | SHA: {sha[:7]}"
    )
    send(webhook_url, msg)


def training_finished(webhook_url, run_name, model, phases, sha,
                      exit_code, elapsed_secs, log_path=None):
    if exit_code == 0:
        icon = "\u2705"
        status = "succeeded"
    else:
        icon = "\u274c"
        status = f"failed (exit {exit_code})"

    elapsed = f" in {elapsed_secs // 60}m" if elapsed_secs is not None else ""

    msg = (
        f"{icon} *{run_name}* {status}{elapsed}\n"
        f"Model: {model} | Phases: {phases} | SHA: {sha[:7]}"
    )

    if exit_code != 0 and log_path:
        try:
            from pathlib import Path
            log = Path(log_path)
            if log.exists():
                tail = "\n".join(log.read_text().splitlines()[-20:])
                msg += f"\n```\n{tail}\n```"
        except Exception:
            pass

    send(webhook_url, msg)
