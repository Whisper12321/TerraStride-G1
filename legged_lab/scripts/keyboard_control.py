from pynput import keyboard

cmd_vx = 0.0
cmd_vy = 0.0
cmd_wyaw = 0.0

def on_press(key):
    global cmd_vx, cmd_vy, cmd_wyaw
    try:
        key_char = getattr(key, "char", None)
        if key == keyboard.Key.up or key_char == "8":
            cmd_vx = 0.75
        elif key == keyboard.Key.down or key_char == "2":
            cmd_vx = -0.75
        elif key == keyboard.Key.left or key_char == "4":
            cmd_vy = 0.5
        elif key == keyboard.Key.right or key_char == "6":
            cmd_vy = -0.5
        elif key == keyboard.Key.shift_l:
            cmd_wyaw = 0.3
        elif key == keyboard.Key.shift_r:
            cmd_wyaw = -0.3
    except AttributeError:
        pass
 
def on_release(key):
    global cmd_vx, cmd_vy, cmd_wyaw
    key_char = getattr(key, "char", None)
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right, keyboard.Key.shift_l, keyboard.Key.shift_r] or key_char in ["8", "2", "4", "6"]:
        cmd_vx = 0.0
        cmd_vy = 0.0
        cmd_wyaw = 0.0

def get_keyboard_commands():
    return cmd_vx, cmd_vy, cmd_wyaw

def keyboard_control():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    return listener
    
