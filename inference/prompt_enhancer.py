"""
Copyright (c) 2025 Tencent. All Rights Reserved.
Licensed under the Tencent Hunyuan Community License Agreement.
"""

import re
import time
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def replace_single_quotes(text):
    """
    Replace single quotes within words with double quotes, and convert
    curly single quotes to curly double quotes for consistency.
    """
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("’", "”")
    replaced_text = replaced_text.replace("‘", "“")
    return replaced_text


class HunyuanPromptEnhancer:

    def __init__(self, models_root_path, device_map="auto"):
        """
        Initialize the HunyuanPromptEnhancer class with model and processor.

        Args:
            models_root_path (str): Path to the pretrained model.
            device_map (str): Device mapping for model loading.
        """
        # Lazy logging setup (will be no-op if already configured by app)
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.model = AutoModelForCausalLM.from_pretrained(
            models_root_path, device_map=device_map, trust_remote_code=True
        )
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            models_root_path, trust_remote_code=True
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot,
        sys_prompt="你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求：1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； 2 在宏观上遵循“总-分-总”的结构，确保信息的层次清晰；3 客观中立，避免主观臆断和情感评价；4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；6 结尾点题，必须用一句话总结图像的整体风格或类型。",
        temperature=0,
        top_p=1.0,
        max_new_tokens=512,
        use_cache=True,
    ):
        """
        Generate a rewritten prompt using the model.

        Args:
            prompt_cot (str): The original prompt to be rewritten.
            sys_prompt (str): System prompt to guide the rewriting.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            max_new_tokens (int): Maximum number of new tokens to generate.
            use_cache (bool): Whether to enable KV cache during generation.

        Returns:
            str: The rewritten prompt, or the original if generation fails.
        """
        org_prompt_cot = prompt_cot
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": org_prompt_cot},
            ]
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,  # Toggle thinking mode (default: True)
            )
            inputs = tokenized_chat.to(self.model.device)
            do_sample = temperature is not None and float(temperature) > 0
            outputs = self.model.generate(
                inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
                use_cache=bool(use_cache),
            )

            # Decode only new tokens and skip special tokens
            generated_sequence = outputs[0]
            prompt_length = inputs.shape[-1]
            new_tokens = generated_sequence[prompt_length:]
            output_res = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            answer_pattern = r"<answer>(.*?)</answer>"
            answer_matches = re.findall(answer_pattern, output_res, re.DOTALL)
            if answer_matches:
                prompt_cot = answer_matches[0].strip()
            else:
                output_clean = re.sub(r"<think>[\s\S]*?</think>", "", output_res)
                output_clean = output_clean.strip()
                prompt_cot = output_clean if output_clean else org_prompt_cot
            prompt_cot = replace_single_quotes(prompt_cot)
            self.logger.info("Re-prompting succeeded; using the new prompt")
        except Exception as e:
            prompt_cot = org_prompt_cot
            self.logger.exception("Re-prompting failed; using the original prompt")

        return prompt_cot


if __name__ == "__main__":
    prompt_enhancer_cls = HunyuanPromptEnhancer(
        models_root_path="/path/to/your/model"
    )

    test_list_zh = [
        "第三人称视角，赛车在城市赛道上飞驰，左上角是小地图，地图下面是当前名次，右下角仪表盘显示当前速度。",
        "第三人称视角，玩家控制一名手拿魔杖的法师与一只巨龙，在一个宽广的平台上战斗。界面左上角上方有角色红色的血条和蓝色的魔法条，右下角显示3个技能图标。",
        "2D横版游戏，左侧是一名手拿武士刀的武士，右侧是一个手拿禅杖的僧人，背景是雾气弥漫的竹林。",
        "第一人称视角，在丛林里用机关枪射击霸王龙，写实风格。",
        "8bit横板格斗游戏，一个蓝色短裤秃头大汉和一个红短裤黄发大汉在进行拳击，屏幕上方是两位玩家的生命值和怒气值，2D游戏，全景。",
        "横板过关游戏，一个紫色衣服的忍者跳在空中，右边是一条鳄鱼，2D游戏，全景。",
        "系列表情包设计。\n表情1：一只猴子咧着嘴笑，下方写着“Happy”。\n表情2：猴子戴着墨镜，下方写着“Cool”。\n表情3：猴子拿着一朵花，表情羞涩，下方写着“Shy”。\n表情4：猴子表情惊讶，下方写着“Surprise”。",
        "一组像素风格武器图标，包括：一柄剑，一把弓和一本魔法书，金属反光感。",
        "设计一套包含9个不同表情的小狗的表情包。",
        "给我生成一张手机壁纸，阳光飘窗前有一盆兰花，不要有窗帘。",
        "韩系插画风女生头像，粉紫色短发+透明感腮红，侧光渲染。",
        "点彩派，盛夏海滨，两位渔夫正在搬运木箱，三艘帆船停在岸边，对角线构图。",
        "一幅由梵高绘制的梦境麦田，旋转的蓝色星云与燃烧的向日葵相纠缠。画面主色调是黄色和蓝色，有铬黄与钴蓝的笔触漩涡。",
        "三只毛绒绒的兔子，其中两只窝在树洞里，另一只在洞外，水彩画风格。",
        "岩彩绘画，渐变半透明玻璃熔体，禅意，中式山水，中国传统配色，装饰画，朱砂红、石青、石绿、土黄等，搭配上金色。",
        "一幅巴洛克风格油画，刻画了一位战神，战神束着铁灰色的长辫，身披由雷电构成的斗篷，他伫立在破碎的罗马柱前，全身视角。",
        "一位二十岁的英国男子在拉小提琴，他拥有蓝色眼睛和金色的头发，中景，由达芬奇绘制的油画。",
        "浮世绘作品，一位妙龄少妇穿着和服，一手打伞一手提一个食盒，走在小桥上，全景。",
        "一位身着皮衣的老人单膝跪雪地上，身旁是一只银白色的狐，身后是结冰的湖面，画面是油画风格。",
        "油画风格，六翼天使，金色头发，张开双臂，手拿权杖，漂浮空中，对称构图，全景镜头，宗教氛围。",
    ]

    test_list_en = [
        "Create a painting depicting a 30 - year - old white female white - collar worker on a business trip by plane. She is sitting next to the window in the business class. Her slim figure is set off by the black suit. Outside the window is the magnificent mountain scenery. The picture uses a frame - style composition and is in the CG animation style.",
        "Depicted in the anime style of Studio Ghibli, Brigitte Lin stands quietly at the edge of the Titanic's deck with a gentle smile on her face. Her figure is graceful, and her eyes are calmly gazing at the vast, blue, sparkling, and vibrant Atlantic Ocean.",
        "A 35 - year - old Asian woman in a professional suit and wearing a delicate watch is standing in front of the big screen at an academic seminar, presenting her views on female entrepreneurship. She is holding a page - turning pen in her hand. The scene is presented in realistic photography style.",
        "Generate an image where the thin Nezha is struggling to climb an extremely huge tree with a canopy like an umbrella. His arms are tightly hugging the trunk, and his feet are also firmly wrapped around it. Around him is a vast forest with uneven trees and quite lush branches and leaves, containing towering ancient giant trees and relatively short but still green grass. The picture is in an upward perspective and presented in the anime style.",
        "Blue background, a lone girl gazes into the distant sea; her expression is sorrowful. Motion blur. In the foreground, a gull glides past; the shot is framed at long range, everything soft and hazy. The mood is mournful yet calm, rendered in cold, dark tones with a film-like grain and a chilly detachment. The image is presented in a realistic photography style.",
        "In the bamboo forest at night, a piece of emerald bamboo was long and slender. The turquoise bamboo leaves glisten in the moonlight. The scene is rendered in realistic photography style.",
        "Painting a picture that contains hills, grass, trees, kites, and a blue sky, with soft and fresh colors, in the impressionist Monet style.",
        "A blend of expressionist and vintage styles, drawing a building with walls made of a patchwork of various contrasting colors, with an orange roof, yellow windows, and a purple door, in a unique overall style, shot at a tilted angle.",
        "A height of about two meters five, large crystal polar bear statue, the whole body is transparent and shiny, every minute detail is very clear. Its silhouette is like the extension of a frozen lake, the bear's body is broad and heavy, like the head, limbs and other places of the carving of the right proportions, to bring people an extremely shocking visual experience. The image has a realistic photography style.",
        "Depicts a group of cocci generating spores, about thirty of them, with rounded spores that possess extraordinary resistance to pressure. The image is presented in an anime style.",
        "Paint a winter scene with crystalline ice hangings hanging from the top of an Antarctic research station and flawless snowflakes piling up on the ground. The image is presented in two dimensional style.",
        "Please draw a picture of the Golden Gate Bridge and the Fisherman's Wharf side by side, using a bird 's-eye view and warm tones as the primary palette. The entire picture is presented in realistic photography style.",
        "Please create an anime-style avatar for me: a magical girl resembling Sailor Moon, looking about 18 or 19 years old, smiling at the camera, with a blurred background.",
        "Realistic photography style, a cuttlefish swims over the sandy seabed near a coral reef, its tentacles gently swaying.",
        "Create a retro-futuristic artwork with the theme of interstellar exploration, featuring elements such as spaceships and lightsabers. The spaceship should include detailed components like space hatches.",
        "Generate a WeChat profile picture in realistic photography style: a curly long-haired girl leaning against a lush green tree, quietly basking in the sunlight. Behind the tree are majestic mountains and water.",
        "Draw a portrait capturing the 35-year-old actor Jackie Chan. He is holding a paintbrush and focused on creating at a drawing board. The image is presented in a vintage realistic photography style.",
        "Draw a pixel-style illustration based on a super cool 25-year-old European woman. She has furrowed brows, long pink hair with slightly curled tips, her mouth slightly open, wearing a black leather jacket and blue jeans.",
        "In a cave piled up with bright gemstones, the gems shimmer with colorful radiance, illuminating the cave brilliantly. Inside the cave, there flows a colorful stream formed by magical spiritual liquid, in which koi fish swim, along with various strange creatures such as elves and phoenixes. At the end of the cave stands a pavilion built of beautiful jade, where a profound white-haired warlock resides. Although the main colors of the scene are different, the tones are unified and harmonious. The painting is presented in an oil painting style.",
        "Depict a scene where a plump witch in a gorgeous dress and a slender elf wearing a pointed hat embrace and dance under the moonlight. The two have a striking contrast in size, like a dream. Hand-painted by a professional artist, presented in a panoramic view with detailed outlines, in a photorealistic style.",
    ]

    for item in test_list_zh:
        print("User Prompt:", item)
        print("---------:")
        time_start = time.time()
        print("RePrompt:", prompt_enhancer_cls.predict(item))
        time_end = time.time()
        print("~~~~~~~~~~~~~~", time_end - time_start)
