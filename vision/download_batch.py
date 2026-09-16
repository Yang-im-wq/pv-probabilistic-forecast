# -*- coding: utf-8 -*-
"""NSMC 风云四号 FY-4B 数据批量下载脚本（正式版）。

两种模式任选其一：

  [模式A] 自动签名下载（推荐，需你的 SecretKey，一次配好全自动）
     python download_batch.py --ak 你的AccessKeyId --sk 你的SecretKey \
         --start 20260901 --end 20260912 --product FDI
     # product: FDI = L1 云图(.HDF) ; CLM = 云检测产品(.NC)
     # 会自动遍历每天 00~23 时，逐个签名下载；已存在/404 自动跳过

  [模式B] 从网站导出的链接列表下载（不需要 SecretKey）
     # 在 NSMC 网站勾选数据，把所有下载链接复制到一个 txt（每行一条）
     python download_batch.py --links links.txt

说明：
  - 下载自动分流：L1 FDI -> data/l1/ ，L2 CLM -> data/clm/
  - 签名算法按阿里云 OSS V1 实现；若 NSMC 不是标准 OSS 会 403，
    此时改用模式B（链接列表）即可。
"""
import os
import sys
import time
import hmac
import hashlib
import base64
import argparse
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "http://clouddata.nsmc.org.cn:8089"
BUCKET = "DATA"

# 产品定义
PRODUCTS = {
    "FDI": {
        "prefix": "FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_",
        "suffix": "_4000M_V0001.HDF",
        "path_tpl": "FY4/FY4B/AGRI/L1/FDI/DISK/4000M/{yyyy}/{yyyymmdd}/{fname}",
        "outdir": r"D:\pv-probabilistic-forecast\data\l1",
        "marker": "_L1-_FDI-",
    },
    "CLM": {
        "prefix": "FY4B-_AGRI--_N_DISK_1050E_L2-_CLM-_MULT_NOM_",
        "suffix": "_4000M_V0001.NC",
        "path_tpl": "FY4/FY4B/AGRI/L2/CLM/DISK/NOM/{yyyy}/{yyyymmdd}/{fname}",
        "outdir": r"D:\pv-probabilistic-forecast\data\clm",
        "marker": "_L2-_CLM-",
    },
}


def oss_sign(secret_key, object_key, expires):
    """阿里云 OSS V1 签名（GET 请求）。"""
    string_to_sign = "GET\n\n\n%d\n/%s/%s" % (expires, BUCKET, object_key)
    h = hmac.new(secret_key.encode(), string_to_sign.encode(), hashlib.sha1)
    return base64.b64encode(h.digest()).decode()


def signed_url(ak, sk, object_key, expires):
    sig = oss_sign(sk, object_key, expires)
    sig_q = urllib.parse.quote(sig, safe="")
    return "%s/%s/%s?AccessKeyId=%s&Expires=%d&Signature=%s" % (
        HOST, BUCKET, object_key, ak, expires, sig_q)


def gen_file_list(product, start, end):
    """生成 [start, end] 日期范围内每天 00~23 时的文件名列表。"""
    files = []
    d = datetime.strptime(start, "%Y%m%d")
    d_end = datetime.strptime(end, "%Y%m%d")
    while d <= d_end:
        yyyy = d.strftime("%Y")
        yyyymmdd = d.strftime("%Y%m%d")
        for hh in range(24):
            ts0 = "%s%02d0000" % (yyyymmdd, hh)
            ts1 = "%s%02d1459" % (yyyymmdd, hh)
            fname = PRODUCTS[product]["prefix"] + ts0 + "_" + ts1 + PRODUCTS[product]["suffix"]
            obj = PRODUCTS[product]["path_tpl"].format(yyyy=yyyy, yyyymmdd=yyyymmdd, fname=fname)
            files.append((fname, obj))
        d += timedelta(days=1)
    return files


def download(url, outdir, fname, retries=5):
    out = os.path.join(outdir, fname)
    # 完整文件大小：L1(~68MB)、CLM(~3MB)；没下完的不算已存在，会重下
    min_size = 50_000_000 if fname.endswith(".HDF") else 2_000_000
    if os.path.exists(out) and os.path.getsize(out) > min_size:
        return "skip"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as f:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return "ok"
        except urllib.error.HTTPError as e:
            # 429 限流：加长退避重试；404 直接放弃
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return "fail:%d" % e.code
        except Exception:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return "fail:err"
    return "fail:timeout"


def main():
    ap = argparse.ArgumentParser(description="NSMC FY-4B 批量下载")
    ap.add_argument("--ak", help="AccessKeyId（模式A）")
    ap.add_argument("--sk", help="SecretKey（模式A）")
    ap.add_argument("--start", help="起始日期 YYYYMMDD，如 20260901")
    ap.add_argument("--end", help="结束日期 YYYYMMDD，如 20260912")
    ap.add_argument("--product", choices=list(PRODUCTS), help="FDI 或 CLM")
    ap.add_argument("--links", help="链接列表 txt（模式B，每行一条 URL）")
    ap.add_argument("--workers", type=int, default=2, help="并发线程数（NSMC会限流，建议2，默认2）")
    args = ap.parse_args()

    if args.links:
        # 模式B：读链接列表
        urls = [l.strip() for l in open(args.links, encoding="utf-8") if l.strip()]
        print("模式B：读链接列表，共 %d 条" % len(urls))
        jobs = []
        for u in urls:
            fname = u.split("?")[0].rstrip("/").split("/")[-1]
            outdir = PRODUCTS["FDI"]["outdir"] if "_L1-_FDI-" in fname else \
                     PRODUCTS["CLM"]["outdir"] if "_L2-_CLM-" in fname else \
                     r"D:\pv-probabilistic-forecast\data\other"
            jobs.append((u, outdir, fname))
    elif args.ak and args.sk and args.start and args.end and args.product:
        # 模式A：自动签名
        files = gen_file_list(args.product, args.start, args.end)
        print("模式A：%s %s~%s 共 %d 个时次" % (args.product, args.start, args.end, len(files)))
        expires = int(time.time()) + 86400 * 7
        outdir = PRODUCTS[args.product]["outdir"]
        jobs = []
        for fname, obj in files:
            jobs.append((signed_url(args.ak, args.sk, obj, expires), outdir, fname))
    else:
        ap.print_help()
        sys.exit("参数不完整：模式A需 --ak --sk --start --end --product；模式B需 --links")

    ok = fail = skip = miss = 0
    total = len(jobs)
    for _, outdir, _ in jobs:
        os.makedirs(outdir, exist_ok=True)

    # 多线程并发下载
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(download, url, outdir, fname): fname for url, outdir, fname in jobs}
        for i, fut in enumerate(as_completed(fut_map), 1):
            r = fut.result()
            fname = fut_map[fut]
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            elif r.startswith("fail:404"):
                miss += 1  # 数据不存在，正常跳过
            else:
                fail += 1
                print("  [%d] 失败(%s) %s" % (i, r, fname), flush=True)
            if i % 20 == 0 or i == total:
                print("进度 %d/%d：成功 %d 跳过 %d 缺失404 %d 失败 %d" %
                      (i, total, ok, skip, miss, fail), flush=True)

    print("完成：成功 %d 跳过 %d 缺失404 %d 失败 %d" % (ok, skip, miss, fail))


if __name__ == "__main__":
    main()
