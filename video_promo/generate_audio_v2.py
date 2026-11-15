import subprocess
import os

scripts = {
    "zh-tw": {
        "voice": "zh-TW-HsiaoChenNeural",
        "rate": "+2%",
        "text": "你是不是也遇過這種困擾？想跑一個 70B 的大模型，光是下載和解壓縮，硬碟就先被吃掉三百多 G，很多筆電就這樣直接卡死。而 LLMZIP 改變了這件事。它讓你在下載的同時直接完成量化，中間完全不產生任何臨時檔案，只佔用最終模型的大小。即使網路斷線，也能零浪費即時續傳；加上密碼學校驗鎖，不用擔心權重損壞。在一般的筆電上，也能優雅運行。想徹底釋放硬碟空間，現在就上 GitHub 搜尋 LLMZIP 吧。"
    },
    "zh-cn": {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+2%",
        "text": "你是不是也遇到过这种困扰？想跑一个 70B 的大模型，光是下载和解压，硬盘就先被吃掉三百多 G，很多笔记本就直接卡死。而 LLMZIP 改变了这件事。它让您在下载的同时直接完成量化，中间完全不产生任何临时文件，只占用最终模型的大小。即使网络中断，也能零浪费即时续传；加上密码学校验锁，不用担心权重损坏。在普通笔记本上，也能优雅运行。想彻底释放硬盘空间，现在就去 GitHub 看看 LLMZIP 吧。"
    },
    "en": {
        "voice": "en-US-AndrewMultilingualNeural",
        "rate": "+0%",
        "text": "Have you ever tried downloading a 70B model, only to watch your disk run completely out of space? Traditional workflows can easily demand over 300 gigabytes of storage. LLMZIP solves this with an elegant approach. By streaming and quantizing directly in memory, it generates zero temporary files, keeping your peak footprint strictly bounded to the final output size. If your connection drops, it resumes instantly with zero wasted bandwidth. It runs smoothly on everyday laptops with no GPU required. Check out LLMZIP on GitHub, and reclaim your disk space today."
    },
    "ja": {
        "voice": "ja-JP-NanamiNeural",
        "rate": "+2%",
        "text": "70Bの大規模モデルを動かそうとして、ストレージ不足に悩まされたことはありませんか？従来のやり方では、一時ファイルだけで300GB以上も消費してしまいます。そこで生まれたのが、LLMZIPです。ダウンロードしながらメモリ上で直接量子化を行うため、一時ファイルは完全にゼロ。途中で回線が切れても、ダウンロード済みのデータを無駄にせず即座に再開できます。高価なGPUがなくても、ノートPCで軽快に動作します。GitHubでLLMZIPをチェックして、ストレージの制約から解放されましょう。"
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
    print(f"Generating naturally [{lang}] with {cfg['voice']} (rate={cfg['rate']})...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(out_path):
        print(f" -> OK: {out_path} ({os.path.getsize(out_path)} bytes)")
    else:
        print(f" -> ERROR [{lang}]: {res.stderr}")
