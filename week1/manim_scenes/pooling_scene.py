import numpy as np
from manim import (
    BLUE_E, TEAL_E, YELLOW, WHITE, TEAL, Scene,
    Text, Write, FadeIn, VGroup, Square, Integer, SurroundingRectangle,
    UP, LEFT, RIGHT, Create, Transform, FadeOut
)


class PoolingScene(Scene):
    def construct(self):
        matrix_values = [
            [4, 3, 8, 2],
            [9, 1, 5, 0],
            [2, 6, 1, 9],
            [5, 3, 2, 4]
        ]
        
        input_color = BLUE_E
        output_color = TEAL_E
        highlight_color = YELLOW
        text_color = WHITE
        
        title = Text("Max Pooling with s=2, f=2", font_size=40).scale(0.6).to_edge(UP)
        self.play(FadeIn(title))

        input_group = VGroup()
        input_squares = [] 
        input_numbers = [] 
        
        for r in range(4):
            row_squares = []
            row_numbers = []
            for c in range(4):
                square = Square(side_length=1.0, fill_color=input_color, fill_opacity=0.5, stroke_color=WHITE)
                square.move_to(np.array([c - 1.5, 1.5 - r, 0]) * 1.0 + LEFT * 3)
                
                num = Integer(matrix_values[r][c], color=text_color)
                num.move_to(square.get_center())
                
                input_group.add(square, num)
                row_squares.append(square)
                row_numbers.append(num)
            input_squares.append(row_squares)
            input_numbers.append(row_numbers)

        input_label = Text("Input Matrix", font_size=24).next_to(input_group, UP)

        output_group = VGroup()
        output_squares = []
        
        for r in range(2):
            row_squares = []
            for c in range(2):
                square = Square(side_length=1.0, fill_color=output_color, fill_opacity=0.5, stroke_color=WHITE)
                square.move_to(np.array([c - 0.5, 0.5 - r, 0]) * 1.0 + RIGHT * 3)
                
                output_group.add(square)
                row_squares.append(square)
            output_squares.append(row_squares)

        output_label = Text("Pooled Output", font_size=24).next_to(output_group, UP)

        self.play(
            FadeIn(input_group), 
            Write(input_label),
            FadeIn(output_group),
            Write(output_label)
        )
        self.wait(0.5)

        regions = [(0,0), (0,2), (2,0), (2,2)]
        targets = [(0,0), (0,1), (1,0), (1,1)]

        kernel_rect = None

        for i, ((r_start, c_start), (out_r, out_c)) in enumerate(zip(regions, targets)):
            top_left_sq = input_squares[r_start][c_start]
            bottom_right_sq = input_squares[r_start+1][c_start+1]
            
            current_kernel_shape = SurroundingRectangle(
                VGroup(top_left_sq, bottom_right_sq),
                color=highlight_color,
                buff=0.1,
                stroke_width=6
            )

            if i == 0:
                kernel_rect = current_kernel_shape
                self.play(Create(kernel_rect), run_time=0.7)
            else:
                self.play(Transform(kernel_rect, current_kernel_shape), run_time=0.7, path_arc=0)
            
            local_values = []
            local_mob_coords = []
            for r in range(r_start, r_start + 2):
                for c in range(c_start, c_start + 2):
                    local_values.append(matrix_values[r][c])
                    local_mob_coords.append((r, c))
            
            max_val = max(local_values)
            max_index = local_values.index(max_val)
            max_r, max_c = local_mob_coords[max_index]
            max_number_mob = input_numbers[max_r][max_c]
            
            self.play(
                max_number_mob.animate.scale(1.5).set_color(highlight_color),
                run_time=0.3
            )
            
            moving_num = Integer(max_val, color=highlight_color, font_size=40)
            moving_num.move_to(max_number_mob.get_center())
            target_square = output_squares[out_r][out_c]
            
            self.play(
                Transform(moving_num, moving_num.copy().move_to(target_square.get_center()).set_color(WHITE)),
                max_number_mob.animate.scale(1/1.5).set_color(text_color), 
                run_time=0.8
            )
            self.add(moving_num) 
            self.wait(0.2)

        self.play(FadeOut(kernel_rect), run_time=0.5)

        final_rect = SurroundingRectangle(output_group, color=TEAL, buff=0.2)
        self.play(Create(final_rect))
        self.wait(3)
