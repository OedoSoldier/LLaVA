
#!/bin/bash
 
# 设置倒计时的总秒数
seconds=60
 
# 倒计时循环
for i in $(seq $seconds -1 1); do
    # 清除之前的输出并打印当前倒计时
    # echo -ne "$i\033[0K\r"
    sleep 1
done

# conda activate llava
# NCCL_DEBUG=INFO nohup bash finetune_lora.sh > 4.log.out 2>&1 &