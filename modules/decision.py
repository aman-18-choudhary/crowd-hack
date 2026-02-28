# modules/decision.py

RISK_THRESHOLD = 20  # set low for demo (change later)


class RiskController:
    def __init__(self):
        self.alert_active = False

    def evaluate(self, current_count, predicted_count):
        risk = False

        if current_count >= RISK_THRESHOLD:
            risk = True

        if predicted_count is not None and predicted_count >= RISK_THRESHOLD:
            risk = True

        if risk and not self.alert_active:
            self.trigger_alert()
            self.alert_active = True

        if not risk and self.alert_active:
            self.reset_alert()
            self.alert_active = False

        return risk

    def trigger_alert(self):
        print("🚨 RISK DETECTED – ACTIVATING SAFETY SYSTEM")

        self.activate_led()
        self.activate_buzzer()
        self.unlock_servo()

    def reset_alert(self):
        print("✅ Crowd Normal – System Reset")

    # --- FAKE IoT SIMULATION FUNCTIONS ---

    def activate_led(self):
        print("🔴 LED PATHWAY ACTIVATED")

    def activate_buzzer(self):
        print("🔊 BUZZER ALERT SOUNDING")

    def unlock_servo(self):
        print("🚪 EMERGENCY GATE UNLOCKED (Servo Triggered)")