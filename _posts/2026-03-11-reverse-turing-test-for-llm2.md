---
title: LLM 的反向图灵测试，与量子力学量子态塌缩
author: cotes
date: 2026-03-11 00:34:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
---

The [favicons](https://www.favicon-generator.org/about/) of [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/) are placed in the directory `assets/img/favicons/`{: .filepath}. You may want to replace them with your own. The following sections will guide you to create and replace the default favicons.


The [favicons](https://www.favicon-generator.org/about/) of [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/) are placed in the directory `assets/img/favicons/`{: .filepath}. You may want to replace them with your own. The following sections will guide you to create and replace the default favicons.

大语言模型可能像一面镜子，实际反映的是使用者的智力水平，这构成了一种反向图灵测试。

传统图灵测试：通过自然语言对话判断机器是否具备人类智能（人类无法区分机器与真人）。

反向图灵测试：通过观察人类的反应来评估人类的智能水平。大语言模型在交互中会"映射"人类的思维特征，对话者的思维深度、提示质量越高，模型表现出的智能水平越显著。

提出好问题的能力尤其重要，同一个现象，不同人，提出的问题就不一样，模型所展现的智能也不一样，由此可以算作反映人的智能，也就是反向图灵测试。

自回归 LM 与扩散 LM？

## Generate the favicon

Prepare a square image (PNG, JPG, or SVG) with a size of 512x512 or more, and then go to the online tool [**Real Favicon Generator**](https://realfavicongenerator.net/) and click the button <kbd>Pick your favicon image</kbd> to upload your image file.

In the next step, the webpage will show all usage scenarios. You can keep the default options, scroll to the bottom of the page, and click the button <kbd>Next →</kbd> to generate the favicon.

## Download & Replace

Download the generated package, unzip and delete the following file(s) from the extracted files:

- `site.webmanifest`{: .filepath}

And then copy the remaining image files (`.PNG`{: .filepath}, `.ICO`{: .filepath} and `.SVG`{: .filepath}) to cover the original files in the directory `assets/img/favicons/`{: .filepath} of your Jekyll site. If your Jekyll site doesn't have this directory yet, just create one.

The following table will help you understand the changes to the favicon files:

| File(s) | From Online Tool | From Chirpy |
| ------- | :--------------: | :---------: |
| `*.PNG` |        ✓         |      ✗      |
| `*.ICO` |        ✓         |      ✗      |
| `*.SVG` |        ✓         |      ✗      |


<!-- markdownlint-disable-next-line -->
>  ✓ means keep, ✗ means delete.
{: .prompt-info }

The next time you build the site, the favicon will be replaced with a customized edition.
