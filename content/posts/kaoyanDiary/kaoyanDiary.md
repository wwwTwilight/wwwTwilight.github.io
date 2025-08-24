---
date: '2025-08-24T15:50:37+08:00'
draft: false
title: '考研日记'
categories : ["考研" , "学习" , "生活"]
tags : ["考研"]
---

记录11408考研的点点滴滴，尽量每一天都会更新，以此来记录一些重要的信息，同时也为自己提供一个反思和总结的空间，也可以当作监督自己效率的工具。

<style>
.countdown-container {
  background: #444c55;
  color: #fff;
  border-radius: 18px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.18), 0 1.5px 8px rgba(80,80,80,0.10);
  padding: 28px 18px 18px 18px;
  max-width: 520px;
  min-width: 320px;
  margin: 36px auto 28px auto;
  text-align: center;
  font-family: 'JetBrains Mono', 'Fira Mono', 'Consolas', 'Menlo', monospace;
  font-size: 1.08rem;
  letter-spacing: 1.1px;
  border: 1.5px solid #888;
  position: relative;
  overflow: hidden;
  transition: box-shadow 0.2s;
  animation: fadeIn 1.2s cubic-bezier(.4,0,.2,1);
}
.countdown-container h2 {
  margin-top: 0;
  margin-bottom: 16px;
  font-size: 1.18em;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-shadow: 0 2px 8px rgba(0,0,0,0.18);
  background: linear-gradient(90deg, #fff 60%, #bbb 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.flip-countdown {
  display: flex;
  justify-content: center;
  gap: 0.7em;
  margin-top: 10px;
}
.flip-block {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.flip-label {
  font-size: 0.7em;
  opacity: 0.85;
  margin-top: 0.1em;
  letter-spacing: 1px;
}
.flip-num {
  background: #222c33;
  color: #fff;
  font-family: 'JetBrains Mono', 'Fira Mono', 'Consolas', 'Menlo', monospace;
  font-size: 2.1em;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  min-width: 2.2em;
  padding: 0.15em 0.3em 0.12em 0.3em;
  border-radius: 0.25em;
  box-shadow: 0 2px 8px rgba(0,0,0,0.18);
  margin-bottom: 2px;
  position: relative;
  perspective: 120px;
  transition: background 0.2s;
  will-change: transform;
}
.flip-animate {
  animation: flipDown 0.5s cubic-bezier(.4,0,.2,1);
}
@keyframes flipDown {
  0% { transform: rotateX(0deg); }
  40% { transform: rotateX(-90deg); }
  60% { transform: rotateX(-90deg); }
  100% { transform: rotateX(0deg); }
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(30px) scale(0.98); }
  to { opacity: 1; transform: none; }
}
</style>
<div class="countdown-container">
  <h2>按照2026年12月23日倒计时</h2>
  <div id="flip-countdown" class="flip-countdown">
    <div class="flip-block"><div class="flip-num" id="flip-days">00</div><div class="flip-label">天</div></div>
    <div class="flip-block"><div class="flip-num" id="flip-hours">00</div><div class="flip-label">小时</div></div>
    <div class="flip-block"><div class="flip-num" id="flip-minutes">00</div><div class="flip-label">分钟</div></div>
    <div class="flip-block"><div class="flip-num" id="flip-seconds">00</div><div class="flip-label">秒</div></div>
  </div>
</div>
<script>
let prevFlip = {d: '', h: '', m: '', s: ''};
function pad2(n) { return n < 10 ? '0' + n : '' + n; }
function flipTo(id, value, key) {
  const el = document.getElementById(id);
  if (!el) return;
  if (prevFlip[key] !== value) {
    el.classList.remove('flip-animate');
    // 触发重绘以重置动画
    void el.offsetWidth;
    el.classList.add('flip-animate');
    el.textContent = value;
    prevFlip[key] = value;
  }
}
function updateFlipCountdown() {
  const target = new Date("2026-12-23 00:00:00").getTime();
  const now = new Date().getTime();
  const distance = target - now;
  if (distance < 0) {
    document.getElementById('flip-countdown').innerHTML = '<span style="font-size:1.5em">🎉 时间到！</span>';
    return;
  }
  const days = Math.floor(distance / (1000 * 60 * 60 * 24));
  const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((distance % (1000 * 60)) / 1000);
  flipTo('flip-days', pad2(days), 'd');
  flipTo('flip-hours', pad2(hours), 'h');
  flipTo('flip-minutes', pad2(minutes), 'm');
  flipTo('flip-seconds', pad2(seconds), 's');
}
updateFlipCountdown();
setInterval(updateFlipCountdown, 1000);
</script>

正经人谁写日记啊？

# 2025-08-24

算是第一天开始复习？虽然什么资料都还没有准备，但是今天首次开始了背单词，不背单词用起来还挺顺手的。剩余的资料还在收集，具体怎么复习尚在规划中，得赶紧找回学习状态。

---

记录一下网上的复习策略

1. 资料这一块
    - 数学：基础方面：针对零基础建议把本科的数学课本拿出来重新看一遍，不过我多少还是记得一点的，不至于零基础，可以看**李永乐复习全书基础篇**，然后是**汤家凤《1800题》**和**李永乐《660题》**，据说然后如果要网课的话就选择**武忠祥**老师的网课，概率论（数一）推荐**余炳森**，线性代数推荐**李永乐**，强化方面：**李林《880题》**强化阶段练习册，**张宇《1000题》**拔高阶段练习册，**李艳芳《真题解析》**前期别碰真题。
    - 英语：app要持续背单词，利用碎片化时间背单词，完整的背单词可以选择**红宝书**，真题方面就选择**黄皮书**，阅读理解可以选择**张剑150篇**。
    - 政治：据说前期不要碰，但是我觉得可以先了解一下，至少先把马原这种不需要有实效性的内容过一遍，然后考前去弄 **《精讲精练》：速看。《1000题》：速刷。《8套卷》and《4套卷》：考前背。**
    - 408：讲义选择王道，参考书考虑这些：严蔚敏《数据结构》、谢希仁《计算机网络》、唐朔飞《计组》、汤子瀛《操作系统》。袁春风据说是原出题组的，他的教材可以当作资料库。
2. 参考学习规划这一块
    - 前期：
    1. 首先准备好数学全书，每天至少3h开始学习数学内容（不论难易，不论耗时，一一攻克）
    2. 早晚各背40min英语单词（拒绝abandon式学习，不要在乎第二天忘不忘）
    3. ~~开始学习C语言基础，包括变量、运算符、循环判断、函数、数组、结构体、指针。主要在于学会格式运用（这几天会发一篇关于C语言基础必备知识点罗列，大家留意一下）~~ 这个对我而言就没有必要了
    4. 建议准备一本天勤数据结构辅导讲义，只看讲义，章节习题要求根据答案做到看懂
    - 基础阶段：
    1. 在继续学习数学的基础上，加练对应章节数学习题（1800、660皆可）。
    2. 继续背英语单词，可在碎片时间内通过单词APP加背。
    3. 准备四本完整的王道辅导讲义，以组成原理20天、数据结构15天、操作系统15天、计算机网络15天的时间将讲义内容快速过一遍，大致建立起各科目的知识框架。（课后习题只做选择，真题不做）
    - 强化阶段：
    1. 数学进入快速刷题阶段，每天上午可以奉献给数学，结合听课+自学+刷题的顺序对数学进行全线的进攻。
    2. 英语单词背的差不多了，此时每天抽出一定的时间加练阅读理解，做完总结。
    3. 专业课方面进入大纲阶段，先结合旧版大纲罗列，可按照大纲查找四本权威教材包含知识点，将教材课后习题根据答案做一遍。
    4. 专业课辅导讲义进入第二遍学习，这一遍在精不在快，需要章节习题除真题外所有题目掌握。
    - 真题阶段：
    1. 数学的二轮开始，巩固基础知识，拔高难点重点，针对薄弱项做专题总结和针对性训练，此时可以速刷1000题和二刷660题，并可开始每两天一套数学真题。
    2. 英语除每天背单词外，可以按时按点刷黄皮书了。
    3. 政治此阶段开始复习，速看全书外，主刷1000题，学后做题，题后反巩固学。
    4. 专业课进入总结阶段，查漏补缺，并开始做真题，按时按点、保质保量的做真题，需要每一道真题都融会贯通。
    - 冲刺阶段：
    1. 每天一套数学卷，真题+模拟卷
    2. 英语除每天固定背单词外，可进行高频单词、同义词、近型词的总结，以及作文模板的专项训练。
    3. 政治每天至少2h的记忆和刷题。
    4. 专业课要进入速背阶段，并且需要做各种较为权威的各类模拟题，平均两天一套。
---

新的mac送过来啦，这之后都是用新的电脑写的啦，之前的M3mac才用了一年，用来以旧换新得到了现在这台m4 16+256，居然只花了1800，ok回到正题

计划我依旧比较模糊，但是我觉得应该先做能做的事情，我打算这几天看看能不能先把复习的资料整理出来，然后数学这一块，概统可以先放一放，我还不太能抓稳是不是考我们学校的硕士，说不定到时候改学校了，换成数二了，免得白忙一场。

明天（其实已经是今天了00:40）有几个任务
- 继续背app单词
- 整理复习资料（27考研版本的可能还没有写出来，可以先用26的）
- 看一下高数和线代的内容吧，就当找回感觉