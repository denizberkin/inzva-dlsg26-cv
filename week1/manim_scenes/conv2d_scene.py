import numpy as np
from manim import (
    Scene, VGroup, Text, Square, Dot,
    BLUE_D, TEAL_D, GOLD_D, YELLOW, WHITE, GRAY_B, TEAL_E,
    RIGHT, UP, ORIGIN,
    FadeIn, FadeOut, TransformFromCopy, Transform, smooth
)


class Conv2DScene(Scene):
    def construct(self):
        np.random.seed(42)
        input_size = 5
        kernel_size = 3
        output_size = input_size - kernel_size + 1
        
        cell_size = 0.60  # Slightly smaller to ensure comfortable horizontal fit
        gap = 0.05
        
        c_input = BLUE_D
        c_kernel = TEAL_D
        c_output = GOLD_D
        c_text = WHITE

        # normalized data with one decimal
        input_data = np.round(np.random.uniform(0, 1, (input_size, input_size)), 1)
        kernel_data = np.round(np.random.uniform(0, 1, (kernel_size, kernel_size)), 1)
        
        # output
        output_data = np.zeros((output_size, output_size))
        for i in range(output_size):
            for j in range(output_size):
                region = input_data[i:i+kernel_size, j:j+kernel_size]
                output_data[i, j] = np.sum(region * kernel_data)
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
            
            grid_group = VGroup(squares, texts)
            grid_group.move_to(ORIGIN)
            
            label = Text(label_text, font_size=22).next_to(grid_group, UP, buff=0.3)
            
            full_object = VGroup(grid_group, label)
            return full_object, squares, texts

        input_full, input_squares, _ = create_grid(input_data, c_input, "Input")
        kernel_full, _, _ = create_grid(kernel_data, c_kernel, "Kernel")
        output_full, output_squares, output_values = create_grid(output_data, c_output, "Feature Map")
        
        output_values.set_opacity(0) # Hide output values initially

        scene_group = VGroup(input_full, kernel_full, output_full).arrange(RIGHT, buff=1.2)
        
        scene_group.move_to(ORIGIN)
        
        self.play(FadeIn(scene_group))
        self.wait(0.5)

        # --- GHOST KERNEL SETUP ---
        original_kernel_grid = kernel_full[0] 
        
        ghost_group = original_kernel_grid.copy()
        ghost_group.set_opacity(0.4) # Semi-transparent
        
        for sq in ghost_group[0]:
            sq.set_stroke(YELLOW, width=3)
            sq.set_fill(TEAL_E, opacity=0.4)
        for txt in ghost_group[1]:
            txt.set_color(YELLOW)

        # 1. Peel off the Ghost Kernel from the main Kernel
        self.play(TransformFromCopy(original_kernel_grid, ghost_group), run_time=0.8)
        
        target_first_cell = input_squares[0]
        ghost_first_cell = ghost_group[0][0] # Top-left square of ghost
        
        shift_vec = target_first_cell.get_center() - ghost_first_cell.get_center()
        
        self.play(ghost_group.animate.shift(shift_vec), run_time=1)
        self.wait(0.2)

        # --- CONVOLUTION LOOP ---
        for i in range(output_size):
            for j in range(output_size):
                target_cell = input_squares[i * input_size + j]
                
                current_ghost_tl = ghost_group[0][0].get_center()
                target_tl = target_cell.get_center()
                move_vec = target_tl - current_ghost_tl
                
                if not (i == 0 and j == 0):
                    run_t = 0.4 if i < 1 else 0.2
                    self.play(
                        ghost_group.animate.shift(move_vec),
                        run_time=run_t,
                        rate_func=smooth
                    )
                
                # 2. Computation Effect (Flash)
                self.play(
                    ghost_group.animate.scale(1.05).set_color(YELLOW).set_opacity(0.8),
                    run_time=0.1
                )
                self.play(
                    ghost_group.animate.scale(1/1.05).set_color(WHITE).set_opacity(0.4), 
                    run_time=0.1
                )
                
                # 3. Send Result to Feature Map
                out_idx = i * output_size + j
                out_sq = output_squares[out_idx]
                out_val = output_values[out_idx]
                
                # A small projectile/dot from Input to Output
                projectile = Dot(color=YELLOW).move_to(ghost_group.get_center())
                self.add(projectile)
                
                self.play(
                    Transform(projectile, out_val),
                    out_sq.animate.set_fill(c_output, opacity=1.0),
                    run_time=0.3
                )
                self.remove(projectile)
                out_val.set_opacity(1)
                self.play(out_sq.animate.set_fill(c_output, opacity=0.7), run_time=0.1)

        # --- FINISH ---
        self.play(FadeOut(ghost_group))
        self.wait(2)