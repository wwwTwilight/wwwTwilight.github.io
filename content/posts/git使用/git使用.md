---
date: '2025-01-15T15:12:36+08:00'
draft: true
title: 'Git使用'
categories: ["学习"]
tags: ["git"]
---

记录一些git常用的指令，主要是防止忘记，内容很乱

# 用户名配置
```
git config --global user.name "用户名"
git config --global user.email 用户邮箱
```
# 同一个origin下设置不同的push和fetch
```
→ study git:(master) git remote add origin git@github.com:git/git

→ study git:(master) git remote set-url --add --push origin git@github.com:MY_REPOSITY/git

→ study git:(master) git remote -v

origin git@github.com:git/git (fetch)

origin git@github.com:MY_REPOSITY/git (push)
```

# git add
将文件加入到暂存区中
```
git add .
```

# git commit
将暂存区的文件提交到本地仓库
```
git commit -m "提交信息"
```

虽然但是，上面两个步骤在vscode里面可以直接提交

# 创建新分支

```
git checkout -b 分支名
```

# tag
创建tag
```
git tag v1.0
```

附注标签
```
git tag -a v1.0 -m "标签信息"
```

# 修改历史版本

```
git commit --amend
```

可以修改最近一次的commit信息

# 清理悬空的commit
```
git fsck --lost-found
```

# 拉取信息

- Clone 拉取完整的仓库到本地目录，可以指定分支，深度。
- Fetch 将远端某些分支最新代码拉取到本地，不会执行 merge 操作，会修改 refs/remote 内的分支信息，如果需要和本地代码合并需要手动操作。
- Pull 拉取远端某分支，并和本地代码进行合并，操作等同于 git fetch + git merge，也可以通过 git pull --rebase 完成 git fetch + git rebase 操作。可能存在冲突，需要解决冲突。

# 合并
merge的两种方式
- Fast-forward：不会产生一个新的merge节点，而是保留原来的历史。
```
git merge 分支名 --ff-only
```
- No-ff：将合并的内容放到一个新的节点。
```
git merge 分支名 --no-ff
```
