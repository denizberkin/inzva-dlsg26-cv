import numpy as np
from manim import (
    Scene, VGroup, Square, Text, Dot,
    BLUE_D, TEAL_D, GOLD_D, WHITE, GRAY_E, GRAY_B, RED_B, YELLOW,
    ORIGIN, UP, RIGHT, DOWN,
    FadeIn, Indicate, Transform, FadeOut, TransformFromCopy
)

class DropoutScene(Scene):
    def construct(self):
        np.random.seed(42)
        input_size = 5
        kernel_size = 3
        output_size = input_size - kernel_size + 1
        dropout_rate = 0.3  # 30% of kernel weights will be dropped
        
        cell_size = 0.60
        gap = 0.05
        
        c_input = BLUE_D
        c_kernel = TEAL_D
        c_output = GOLD_D
        c_text = WHITE
        c_drop = GRAY_E

        # 1. Data Generation
        input_data = np.round(np.random.uniform(0, 1, (input_size, input_size)), 1)
        kernel_data = np.round(np.random.uniform(0, 1, (kernel_size, kernel_size)), 1)
        
        # --- DROPOUT LOGIC ---
        # Create a mask: 1 for keep, 0 for drop
        mask = np.random.choice([0, 1], size=(kernel_size, kernel_size), p=[dropout_rate, 1-dropout_rate])
        dropped_kernel_data = kernel_data * mask # Elements set to 0.0

        # Output calculation based on DROPPED kernel
        output_data = np.zeros((output_size, output_size))
        for i in range(output_size):
            for j in range(output_size):
                region = input_data[i:i+kernel_size, j:j+kernel_size]
                output_data[i, j] = np.sum(region * dropped_kernel_data)
        output_data = np.round(output_data, 1)

        def create_grid(data, color, label_text, font_size=20):
            rows, cols = data.shape
            squares = VGroup()
            texts = VGroup()
            for i in range(rows):
                for j in range(cols):
                    sq = Square(side_length=cell_size)
                    sq.set_fill(color, opacity=0.7)
                    sq.set_stroke(color=GRAY_B, width=1.0)
                    sq.move_to(np.array([j * (cell_size + gap), -i * (cell_size + gap), 0]))
                    val_str = f"{data[i, j]:.1f}"
                    txt = Text(val_str, font_size=font_size, color=c_text)
                    txt.move_to(sq.get_center())
                    squares.add(sq)
                    texts.add(txt)
            
            grid_group = VGroup(squares, texts).move_to(ORIGIN)
            label = Text(label_text, font_size=22).next_to(grid_group, UP, buff=0.3)
            return VGroup(grid_group, label), squares, texts

        # Build UI
        input_full, input_squares, _ = create_grid(input_data, c_input, "Input")
        kernel_full, kernel_squares, kernel_texts = create_grid(kernel_data, c_kernel, "Kernel (Weights)")
        output_full, output_squares, output_values = create_grid(output_data, c_output, "Feature Map")
        
        output_values.set_opacity(0)
        scene_group = VGroup(input_full, kernel_full, output_full).arrange(RIGHT, buff=1.2).move_to(ORIGIN)
        
        self.play(FadeIn(scene_group))
        self.wait(1)

        # --- ANIMATE DROPOUT ON KERNEL ---
        drop_indices = np.where(mask.flatten() == 0)[0]
        drop_anims = []
        for idx in drop_indices:
            # Change color to gray and set text to 0.0 to show it's dropped
            new_txt = Text("0.0", font_size=20, color=RED_B).move_to(kernel_texts[idx].get_center())
            drop_anims.append(kernel_squares[idx].animate.set_fill(c_drop, opacity=0.3))
            drop_anims.append(Transform(kernel_texts[idx], new_txt))

        dropout_label = Text("Applying Dropout", font_size=16, color=RED_B).next_to(kernel_full, DOWN)
        self.play(*drop_anims, FadeIn(dropout_label))
        self.play(Indicate(kernel_full))
        self.wait(1)
        self.play(FadeOut(dropout_label))

        # --- CONVOLUTION WITH DROPPED KERNEL ---
        ghost_group = kernel_full[0].copy().set_opacity(0.4)
        for sq in ghost_group[0]:
            sq.set_stroke(YELLOW, width=3)
        
        self.play(TransformFromCopy(kernel_full[0], ghost_group))
        
        # Initial shift to start position
        shift_vec = input_squares[0].get_center() - ghost_group[0][0].get_center()
        self.play(ghost_group.animate.shift(shift_vec))

        for i in range(output_size):
            for j in range(output_size):
                idx = i * output_size + j
                target_tl = input_squares[i * input_size + j].get_center()
                move_vec = target_tl - ghost_group[0][0].get_center()
                
                if not (i == 0 and j == 0):
                    self.play(ghost_group.animate.shift(move_vec), run_time=0.3)
                
                # Flash effect
                self.play(ghost_group.animate.scale(1.1), run_time=0.1)
                self.play(ghost_group.animate.scale(1/1.1), run_time=0.1)

                # Result projectile
                res_val = output_values[idx]
                projectile = Dot(color=YELLOW).move_to(ghost_group.get_center())
                self.add(projectile)
                self.play(
                    Transform(projectile, res_val),
                    output_squares[idx].animate.set_fill(c_output, opacity=1.0),
                    run_time=0.3
                )
                res_val.set_opacity(1)
                self.remove(projectile)
                self.play(output_squares[idx].animate.set_fill(c_output, opacity=0.7), run_time=0.1)

        self.play(FadeOut(ghost_group))
        self.wait(2)