from typing import Any
import imgui
from pathlib import Path
import uuid

class FileSelectorUI:
    def __init__(self, label: str, ext: str, unique_tag_suffix: str):
        self.label = label
        self.ext = ext
        self.selected_file = None
        self.unique_tag_suffix = unique_tag_suffix

    def __call__(self, root_dir) -> Any:
        changed = False
        if imgui.button(f"select##select_{self.unique_tag_suffix}"):
            imgui.open_popup(f"select-popup##select-popup_{self.unique_tag_suffix}")
        imgui.same_line()
        imgui.text(self.label + ": " + str(self.selected_file))

        if imgui.begin_popup(f"select-popup##select-popup_{self.unique_tag_suffix}"):
            for file in Path(root_dir).rglob("*." + self.ext):
                clicked, selected = imgui.selectable(str(file), selected=(file == self.selected_file))
                if clicked:
                    self.selected_file = file
                    changed = True
            imgui.end_popup()
        return changed

class TrajectorySelectorUI:
    def __init__(self, root_dir: str):
        self.traj_file1 = FileSelectorUI("Trajectory 1", "mp4", "traj1")
        self.traj_file2 = FileSelectorUI("Trajectory 2", "mp4", "traj2")
        self.root_dir = root_dir
        self.cond_scale = 1.0

    def __call__(self) -> Any:
        imgui.begin("Trajectory Selector", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        changed = False
        _changed, self.root_dir = imgui.input_text("Root Dir", self.root_dir)
        changed = changed or _changed
        changed = changed or self.traj_file1(self.root_dir)
        changed = changed or self.traj_file2(self.root_dir)
        _changed, self.cond_scale = imgui.input_float("Cond Scale", self.cond_scale)
        changed = changed or _changed
        imgui.end()

        return changed
