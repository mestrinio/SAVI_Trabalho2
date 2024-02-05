import pyautogui

def screenshot():
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'objects_pcd/objectspng/cena.png')