from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    """
    A custom callback for early stopping.
    Stops training after a specified number of steps.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= 7:
            control.should_training_stop = True
            return control

class PrintLossCallback(TrainerCallback):
    """
    A custom callback to print loss at specified steps.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.log_history:
            print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss']}")
