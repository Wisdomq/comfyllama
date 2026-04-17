"""
Email Notification Script
Sends email when training completes using server's mail service
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import sys
import os

def send_training_complete_email(
    recipient_email,
    training_time_hours=None,
    final_loss=None,
    total_steps=None,
    model_path="./comfyui_lora_model_final",
    smtp_server="localhost",
    smtp_port=25,
    sender_email="training@localhost"
):
    """
    Send email notification that training has completed
    
    Args:
        recipient_email: Email address to send notification to
        training_time_hours: Total training time in hours
        final_loss: Final training loss
        total_steps: Total training steps completed
        model_path: Path where model was saved
        smtp_server: SMTP server address (default: localhost)
        smtp_port: SMTP port (default: 25 for local mail)
        sender_email: Sender email address
    """
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = '✓ ComfyUI Model Training Complete'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    # Create email body
    completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Plain text version
    text_body = f"""
ComfyUI Model Training Completed Successfully!

Completion Time: {completion_time}
Server: AMD EPYC 7402P (48 cores)

Training Summary:
{'='*50}
"""
    
    if training_time_hours:
        text_body += f"Training Duration: {training_time_hours:.2f} hours\n"
    if final_loss:
        text_body += f"Final Loss: {final_loss:.4f}\n"
    if total_steps:
        text_body += f"Total Steps: {total_steps}\n"
    
    text_body += f"""
Model Location: {model_path}

Next Steps:
1. Test the model: python test_model.py
2. Download model from server if needed
3. Deploy to production

{'='*50}

This is an automated notification from your training server.
"""
    
    # HTML version (prettier)
    html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 5px; }}
        .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; border-radius: 5px; }}
        .metric {{ background-color: white; padding: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
        .footer {{ margin-top: 20px; padding: 10px; text-align: center; color: #666; font-size: 12px; }}
        h2 {{ color: #333; }}
        .success {{ color: #4CAF50; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✓ Training Complete!</h1>
            <p>ComfyUI Workflow Generator</p>
        </div>
        
        <div class="content">
            <h2>Training Summary</h2>
            <p><strong>Completion Time:</strong> {completion_time}</p>
            <p><strong>Server:</strong> AMD EPYC 7402P (48 cores)</p>
            
            <div class="metric">
                <strong>Training Duration:</strong> {training_time_hours:.2f} hours
            </div>
            
            <div class="metric">
                <strong>Final Loss:</strong> <span class="success">{final_loss:.4f}</span>
            </div>
            
            <div class="metric">
                <strong>Total Steps:</strong> {total_steps}
            </div>
            
            <div class="metric">
                <strong>Model Location:</strong> {model_path}
            </div>
            
            <h2>Next Steps</h2>
            <ol>
                <li>Test the model: <code>python test_model.py</code></li>
                <li>Download model from server if needed</li>
                <li>Deploy to production</li>
            </ol>
        </div>
        
        <div class="footer">
            <p>This is an automated notification from your training server.</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Attach both versions
    part1 = MIMEText(text_body, 'plain')
    part2 = MIMEText(html_body, 'html')
    msg.attach(part1)
    msg.attach(part2)
    
    # Send email
    try:
        print(f"\nSending email notification to {recipient_email}...")
        
        # Connect to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            # For local mail server, usually no authentication needed
            # If authentication is required, uncomment:
            # server.starttls()
            # server.login(sender_email, password)
            
            server.send_message(msg)
        
        print(f"✓ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        print(f"  SMTP Server: {smtp_server}:{smtp_port}")
        print(f"  Recipient: {recipient_email}")
        return False


if __name__ == "__main__":
    """
    Usage:
    python send_email_notification.py your.email@example.com [training_time] [final_loss] [total_steps]
    
    Example:
    python send_email_notification.py user@example.com 5.5 0.45 1500
    """
    
    if len(sys.argv) < 2:
        print("Usage: python send_email_notification.py <recipient_email> [training_time] [final_loss] [total_steps]")
        print("\nExample:")
        print("  python send_email_notification.py user@example.com 5.5 0.45 1500")
        sys.exit(1)
    
    recipient = sys.argv[1]
    training_time = float(sys.argv[2]) if len(sys.argv) > 2 else None
    final_loss = float(sys.argv[3]) if len(sys.argv) > 3 else None
    total_steps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    print("="*70)
    print("SENDING TRAINING COMPLETION EMAIL")
    print("="*70)
    
    success = send_training_complete_email(
        recipient_email=recipient,
        training_time_hours=training_time,
        final_loss=final_loss,
        total_steps=total_steps
    )
    
    if success:
        print("\n✓ Notification sent successfully!")
    else:
        print("\n✗ Failed to send notification")
        print("\nTroubleshooting:")
        print("1. Check if mail service is running: systemctl status postfix")
        print("2. Check mail logs: tail -f /var/log/mail.log")
        print("3. Test mail command: echo 'test' | mail -s 'test' your@email.com")
    
    print("="*70)
