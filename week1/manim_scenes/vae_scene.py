from manim import (
    Scene, Square, VGroup, Text, MathTex, Polygon, RoundedRectangle,
    GREY_B, TEAL_C, TEAL_E, YELLOW_C, MAROON_B, WHITE, GREY, BLUE_E, BLUE_C, PURPLE_E, PURPLE_C, RED,
    UP, DOWN, RIGHT, 
    FadeIn, Write, Create, FadeOut, Transform, linear,
)

class VAEScene(Scene):
    def construct(self):
        # --- COLOR SCHEME ---
        COL_INPUT = GREY_B
        COL_ENC_BG = BLUE_E
        COL_ENC_PACKET = BLUE_C
        COL_DEC_BG = PURPLE_E
        COL_DEC_PACKET = PURPLE_C
        COL_MU = TEAL_C
        COL_SIGMA = TEAL_E
        COL_NOISE = YELLOW_C
        COL_Z = MAROON_B
        
        # --- COORDINATE DEFINITIONS ---
        enc_x_start, enc_x_end = -3.5, -1.5
        enc_y_tall, enc_y_short = 1.2, 0.5
        
        dec_x_start, dec_x_end = 1.5, 3.5
        dec_y_short, dec_y_tall = 0.5, 1.2
        
        latent_x = 0
        packet_thickness = 0.15

        # --- 1. STATIC ARCHITECTURE SETUP ---
        side_len = 1.5
        input_grid_frame = Square(side_length=side_len).set_color(COL_INPUT).set_fill(COL_INPUT, opacity=0.1)
        pixel_squares = []
        for _ in range(16):
            S = Square(side_length=side_len/4).set_stroke(width=0.5, color=GREY)
            S.set_fill(WHITE, opacity=(len(pixel_squares)%5)/5 * 0.8 + 0.1) 
            pixel_squares.append(S)
            
        input_pixels = VGroup(*pixel_squares).arrange_in_grid(4, 4, buff=0).move_to(input_grid_frame)
        input_group = VGroup(input_grid_frame, input_pixels).move_to([enc_x_start - 1.5, 0, 0])
        input_label = Text("Input x", font_size=24).next_to(input_group, UP)

        encoder_bg = Polygon(
            [enc_x_start, enc_y_tall, 0], [enc_x_end, enc_y_short, 0],
            [enc_x_end, -enc_y_short, 0], [enc_x_start, -enc_y_tall, 0],
            color=COL_ENC_BG, fill_opacity=0.2, stroke_width=2
        )
        enc_label = Text("Encoder", font_size=20, color=COL_ENC_BG).next_to(encoder_bg, UP)

        decoder_bg = Polygon(
            [dec_x_start, dec_y_short, 0], [dec_x_end, dec_y_tall, 0],
            [dec_x_end, -dec_y_tall, 0], [dec_x_start, -dec_y_short, 0],
            color=COL_DEC_BG, fill_opacity=0.2, stroke_width=2
        )
        dec_label = Text("Decoder", font_size=20, color=COL_DEC_BG).next_to(decoder_bg, UP)

        output_group = input_group.copy().move_to([dec_x_end + 1.5, 0, 0])
        for submob in output_group[1]:
             curr_op = submob.get_fill_opacity()
             submob.set_fill(opacity=curr_op * 0.7)
        output_label = Text("Reconstruction x'", font_size=24).next_to(output_group, UP)


        # --- ANIMATION START ---
        self.play(
            FadeIn(input_group), Write(input_label),
            Create(encoder_bg), Write(enc_label),
            Create(decoder_bg), Write(dec_label),
            FadeIn(output_group), Write(output_label)
        )
        self.wait(0.5)

        # flow
        enc_packet_start_shape = Polygon(
            [enc_x_start, enc_y_tall, 0], [enc_x_start + packet_thickness, enc_y_tall, 0],
            [enc_x_start + packet_thickness, -enc_y_tall, 0], [enc_x_start, -enc_y_tall, 0],
            color=COL_ENC_PACKET, fill_opacity=1, stroke_width=0
        )
        enc_packet_end_shape = Polygon(
             [enc_x_end - packet_thickness, enc_y_short, 0], [enc_x_end, enc_y_short, 0],
             [enc_x_end, -enc_y_short, 0], [enc_x_end - packet_thickness, -enc_y_short, 0],
             color=COL_ENC_PACKET, fill_opacity=1, stroke_width=0
        )

        mover_packet = input_group.copy()
        self.add(mover_packet)

        self.play(Transform(mover_packet, enc_packet_start_shape), run_time=0.7)
        
        self.play(Transform(mover_packet, enc_packet_end_shape), run_time=1.2, rate_func=linear)
        
        vec_h, vec_w = 1.5, 0.25
        mu_vec = RoundedRectangle(corner_radius=0.1, height=vec_h, width=vec_w, color=COL_MU, fill_opacity=0.9).move_to([latent_x - 0.5, 0.5, 0])
        sigma_vec = RoundedRectangle(corner_radius=0.1, height=vec_h, width=vec_w, color=COL_SIGMA, fill_opacity=0.9).move_to([latent_x + 0.5, 0.5, 0])
        mu_label_txt = MathTex(r"\mu", color=COL_MU).next_to(mu_vec, UP)
        sigma_label_txt = MathTex(r"\sigma", color=COL_SIGMA).next_to(sigma_vec, UP)
        latent_group_separate = VGroup(mu_vec, sigma_vec, mu_label_txt, sigma_label_txt)

        self.play(Transform(mover_packet, latent_group_separate))
        self.wait(0.3)

        noise_vec = RoundedRectangle(corner_radius=0.1, height=vec_h, width=vec_w, color=COL_NOISE, fill_opacity=0.7).move_to([latent_x, -1.5, 0])
        noise_label_txt = MathTex(r"\epsilon", color=COL_NOISE).next_to(noise_vec, DOWN)
        latent_center = encoder_bg.get_right() + RIGHT * 2.5 + UP * 0.5
        formula = MathTex(r"z = \mu + \sigma \odot \epsilon", font_size=30).move_to(latent_center + UP*2)
        kl_text = Text("KL Divergence Loss", font_size=16, color=RED).next_to(formula, DOWN)

        self.play(FadeIn(noise_vec), Write(noise_label_txt))
        self.play(Write(formula), Write(kl_text))
        self.wait(1.0)

        z_vec = RoundedRectangle(corner_radius=0.1, height=vec_h, width=vec_w, color=COL_Z, fill_opacity=1.0).move_to([latent_x, 0, 0])
        z_label_txt = MathTex("z", color=COL_Z).next_to(z_vec, UP)
        self.play(
            mover_packet.animate.move_to(z_vec.get_center()).set_opacity(0),
            noise_vec.animate.move_to(z_vec.get_center()).set_opacity(0),
            FadeOut(mu_label_txt), FadeOut(sigma_label_txt), FadeOut(noise_label_txt),
            FadeIn(z_vec), Write(z_label_txt),
            run_time=1.0
        )
        self.wait(1.0)
        
        mover_packet = z_vec 

        dec_packet_start_shape = Polygon(
            [dec_x_start, dec_y_short, 0], [dec_x_start + packet_thickness, dec_y_short, 0],
            [dec_x_start + packet_thickness, -dec_y_short, 0], [dec_x_start, -dec_y_short, 0],
            color=COL_DEC_PACKET, fill_opacity=1, stroke_width=0
        )
        dec_packet_end_shape = Polygon(
             [dec_x_end - packet_thickness, dec_y_tall, 0], [dec_x_end, dec_y_tall, 0],
             [dec_x_end, -dec_y_tall, 0], [dec_x_end - packet_thickness, -dec_y_tall, 0],
             color=COL_DEC_PACKET, fill_opacity=1, stroke_width=0
        )

        self.play(
            Transform(mover_packet, dec_packet_start_shape),
            FadeOut(z_label_txt),
            run_time=0.7
        )

        self.play(Transform(mover_packet, dec_packet_end_shape), run_time=1.2, rate_func=linear)
        self.play(Transform(mover_packet, output_group), run_time=0.7)
        self.remove(mover_packet)
        
        rec_text = Text("Reconstruction Loss", font_size=16, color=RED).next_to(output_group, DOWN)
        self.play(Write(rec_text))

        self.wait(2)
