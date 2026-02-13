import numpy as np
from manim import (
    Scene, VGroup, Square, Text, Tex, MathTex, Axes, Arrow, Brace,
    BLUE_D, BLUE_C, BLUE_B, GOLD, RED, GREEN, GOLD_A, WHITE, GRAY, GREY,
    UP, DOWN, LEFT, RIGHT, BOLD, ORIGIN, DEGREES, 
    FadeIn, FadeOut, Write, Create, Transform, ReplacementTransform, GrowFromCenter, GrowArrow, LaggedStart
)

class BatchNorm2DVisual(Scene):
    def construct(self):
        # --- CONFIGURATION & PALETTE ---
        # self.camera.background_color = "#1e1e1e"
        c_batch = [BLUE_D, BLUE_C, BLUE_B]
        c_channel_highlight = GOLD
        
        # --- SECTION 1: TITLE ---
        title = Text("Batch Normalization (2D Conv)", font_size=36, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title), run_time=1.0)

        # --- SECTION 2: TENSOR VISUALIZATION ---
        batch_size = 3
        channels = 3
        
        # Group to hold all maps
        feature_maps = VGroup()
        
        # Create the grid
        for b in range(batch_size):
            batch_row = VGroup()
            for c in range(channels):
                # Feature Map Rectangle
                fmap = Square(side_length=1.2)
                fmap.set_fill(c_batch[b], opacity=0.8)
                fmap.set_stroke(WHITE, width=2)
                
                # Label
                label = MathTex(f"x_{{{b},{c}}}", font_size=24).move_to(fmap.center())
                
                # Combine map and label
                map_group = VGroup(fmap, label)
                batch_row.add(map_group)
            
            batch_row.arrange(RIGHT, buff=0.5)
            feature_maps.add(batch_row)
        
        feature_maps.arrange(DOWN, buff=0.5)
        feature_maps.move_to(ORIGIN).shift(DOWN * 0.5)
        
        # Labels for the grid
        batch_label = Text("Batch (N)", font_size=24, color=GRAY).next_to(feature_maps, LEFT, buff=0.5).rotate(90 * DEGREES)
        channel_label = Text("Channels (C)", font_size=24, color=GRAY).next_to(feature_maps, UP, buff=0.5)

        self.play(
            LaggedStart(*[FadeIn(row, shift=UP) for row in feature_maps], lag_ratio=0.2),
            Write(batch_label),
            Write(channel_label)
        )
        self.wait(1)

        # --- SECTION 3: CHANNEL SELECTION ---
        explanation_1 = Tex(r"Step 1: Select Channel $k$ across the entire Batch", font_size=24)
        explanation_1.set_color_by_tex("$k$", GOLD)
        explanation_1.to_edge(UP).shift(DOWN * 0.6)
        
        # Animations: Dim everything except Column 1 (Channel 1)
        anims = []
        target_maps = VGroup() # Store the highlighted maps to move them later
        
        for b in range(batch_size):
            for c in range(channels):
                m_group = feature_maps[b][c]
                if c == 1: # The selected channel (middle)
                    anims.append(m_group[0].animate.set_fill(c_channel_highlight, opacity=1.0))
                    anims.append(m_group[0].animate.set_stroke(GOLD_A, width=4))
                    target_maps.add(m_group)
                else:
                    anims.append(m_group.animate.set_opacity(0.1))

        self.play(
            FadeOut(title),
            FadeIn(explanation_1),
            *anims
        )
        self.wait(1)

        selected_col = VGroup(*[feature_maps[b][1] for b in range(batch_size)])
        
        self.play(
            FadeOut(batch_label), FadeOut(channel_label),
            # Fade out non-selected maps
            *[FadeOut(feature_maps[b][c]) for b in range(batch_size) for c in range(channels) if c != 1],
            # Move selected to left
            selected_col.animate.arrange(DOWN, buff=0.2).to_edge(LEFT, buff=1.0),
            # explanation_1.
        )
        
        # Add a brace to show these are pooled
        brace = Brace(selected_col, RIGHT)
        brace_text = brace.get_text(r"Compute $\mu, \sigma^2$")
        
        self.play(GrowFromCenter(brace), Transform(explanation_1, brace_text))
        self.wait(0.5)

        # --- SECTION 4: STATISTICS & NORMALIZATION ---
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.5],
            x_length=6,
            y_length=4,
            axis_config={"color": GREY},
        ).to_edge(RIGHT, buff=1.0)
        
        # 1. Unnormalized Distribution (Shifted and Wide)
        mu_start, sigma_start = 1.5, 1.2
        dist_raw = axes.plot(lambda x: np.exp(-0.5 * ((x - mu_start)/sigma_start)**2) / (sigma_start * np.sqrt(2*np.pi)), color=RED)
        label_raw = MathTex(r"\text{Input Distribution } x").next_to(axes, UP).set_color(RED)
        
        self.play(Create(axes), Create(dist_raw), Write(label_raw))
        self.wait(1)

        # 2. Normalize (Center at 0, Scale to 1)
        form_norm = MathTex(r"\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}", font_size=36)
        form_norm.next_to(brace, RIGHT, buff=0.5).shift(UP*1.5)
        
        dist_norm = axes.plot(lambda x: np.exp(-0.5 * x**2) / np.sqrt(2*np.pi), color=GREEN)
        label_norm = MathTex(r"\text{Normalized } \hat{x} \sim N(0,1)").next_to(axes, UP).set_color(GREEN)

        self.play(
            Transform(explanation_1, form_norm),
            Transform(dist_raw, dist_norm),
            Transform(label_raw, label_norm)
        )
        self.wait(1)

        # 3. Scale and Shift (Gamma and Beta)
        # Formula
        form_scale = MathTex(r"y = \gamma \hat{x} + \beta", font_size=36)
        form_scale.move_to(form_norm.get_center())
        
        # Final Distribution (Learned parameters)
        gamma, beta = 0.7, -1.0
        dist_final = axes.plot(lambda x: np.exp(-0.5 * ((x - beta)/gamma)**2) / (gamma * np.sqrt(2*np.pi)), color=GOLD)
        label_final = MathTex(r"\text{Output } y").next_to(axes, UP).set_color(GOLD)
        
        # Add labels for Gamma and Beta arrows
        arrow_beta = Arrow(start=axes.c2p(0, 0.5), end=axes.c2p(beta, 0.5), color=GOLD, buff=0)
        lbl_beta = MathTex(r"\beta").next_to(arrow_beta, UP, buff=0.1).set_color(GOLD)
        
        self.play(
            ReplacementTransform(explanation_1, form_scale), # reuse explanation object for smooth transition
            Transform(dist_raw, dist_final), # dist_raw is actually holding the green curve now due to previous transform
            Transform(label_raw, label_final),
            GrowArrow(arrow_beta),
            Write(lbl_beta)
        )
        self.wait(2)
