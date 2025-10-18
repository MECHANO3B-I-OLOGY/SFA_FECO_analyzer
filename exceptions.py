from tkinter import Toplevel, Label, Button, Checkbutton, IntVar, messagebox

def error_popup(msg=None):
    if msg == None:
        msg = "Uncaught unknown exception has occurred, please view the terminal for details"
    messagebox.showerror("Error", msg)

def warning_popup(msg):
    messagebox.showwarning("Warning", msg)

def warning_prompt(msg):
    return messagebox.askokcancel("Warning", msg)

def checkbox_popup(parent, title, message):
    """
    Show a popup with a message and a 'Don't show this message again' checkbox.
    Returns True if the checkbox is checked, False otherwise.

    Args:
        parent: The parent Tk window or Toplevel window.
        title (str): The popup window title.
        message (str): The message to display.
    """
    popup = Toplevel(parent)
    popup.title(title)
    popup.resizable(False, False)
    popup.transient(parent)   # Keep on top of the parent
    popup.grab_set()          # Modal behavior

    # Message
    Label(popup, text=message, wraplength=300, justify="left", padx=20, pady=10).pack()

    # Checkbox
    var = IntVar()
    Checkbutton(popup, text="Don't show this message again", variable=var).pack(pady=(0, 10))

    # OK button
    Button(popup, text="OK", width=10, command=popup.destroy).pack(pady=(0, 10))

    popup.wait_window()  # Wait until user closes popup
    return bool(var.get())
