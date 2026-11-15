import subprocess
import os
import sys

scripts = {
    "zh-tw": {
        "voice": "zh-TW-YunJheNeural",
        "rate": "+15%",
        "text": "你的硬碟又被開源大模型塞爆了嗎？下載 70B 模型居然要吃掉 300G 磁碟空間！現在，有了 LLMZIP！邊下載、邊量化，零暫存檔，只佔最終模型大小！就算網路斷線，4KB Micro-WAL 讓你一秒接續、零流量浪費！密碼學防毒原子鎖，保證權重絕無損壞。筆電也能跑！立刻上 GitHub 搜尋 LLMZIP，釋放你的硬碟空間！"
    },
    "zh-cn": {
        "voice": "zh-CN-YunxiNeural",
        "rate": "+15%",
        "text": "你的硬盘又被开源大模型塞爆了吗？下载 70B 模型居然要吃掉 300G 磁盘空间！现在，有了 LLMZIP！边下载、边量化，零临时文件，只占最终模型大小！就算网络断线，4KB Micro-WAL 让你一秒接续、零流量浪费！密码学防毒原子锁，保证权重绝无损坏。笔记本也能跑！立刻上 GitHub 搜索 LLMZIP，释放你的硬盘空间！"
    },
    "en": {
        "voice": "en-US-AndrewMultilingualNeural",
        "rate": "+12%",
        "text": "Still wasting 300GB of disk space downloading 70B open-source models? Stop the double-space dilemma! Meet LLMZIP! Stream and quantize on the fly with zero temp files! Peak footprint is only 1.0x output size. Connection dropped? Our 4KB Micro-WAL resumes with zero bytes wasted! Cryptographically locked, hardware safe, no GPU required! Search LLMZIP on GitHub and star the repo today!"
    },
    "ja": {
        "voice": "ja-JP-KeitaNeural",
        "rate": "+15%",
        "text": "70B大規模モデルのダウンロードで、SSD容量が爆発していませんか？もう二重ストレージ問題で悩む必要はありません！革命的ツール、LLMZIP！ダウンロードしながら直接量子化！一時ファイルは完全ゼロ！通信が途切れても、4KBのMicro-WALでデータ無駄ゼロの即時再開！暗号ロックで破損も完全防止！今すぐGitHubでLLMZIPを検索して、Starを押そう！"
    }
}

out_dir = r"e:\Dev\AI\LLMZIP\video_promo\audio"
os.makedirs(out_dir, exist_ok=True)

for lang, cfg in scripts.items():
    out_path = os.path.join(out_dir, f"vo_{lang}.mp3")
    cmd = [
        "edge-tts",
        "--voice", cfg["voice"],
        "--rate", cfg["rate"],
        "--text", cfg["text"],
        "--write-media", out_path
    ]
    print(f"Generating [{lang}] with {cfg['voice']}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(out_path):
        print(f" -> OK: {out_path} ({os.path.getsize(out_path)} bytes)")
    else:
        print(f" -> ERROR [{lang}]: {res.stderr}")
