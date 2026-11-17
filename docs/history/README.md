# 提交署名更正与实验溯源

2026-11-17，按仓库所有者要求，将此前误用助手身份的25条提交统一更正为GitHub账号 **PLUTO-SCY**。作者与提交者均使用账号公开名称`chenyang shao`及GitHub隐私邮箱`78310004+PLUTO-SCY@users.noreply.github.com`。当前checkout的仓库局部Git配置也已更正；不改系统或其他仓库的配置。

只更正历史提交元数据及其父提交引用。25条历史提交各自的文件树、提交说明、作者/提交者日期全部保留。代码、数据、评分、曲线和正负实验结果没有因署名修正而改变。完整旧历史另保存为本地Git bundle备份，未推送旧署名备份分支。

## 如何使用旧实验记录中的提交ID

Git提交ID包含作者、提交者和父提交信息，因此本次更正会改变历史SHA。旧实验JSON仍保留原来的SHA，不回写原始证据；对应的源码可通过[逐条映射表](identity-rewrite-map.json)定位。表中每对旧/新提交的`tree`相同，表示该提交的文件内容完全一致。

在当前仓库根目录运行，例如查询旧强起点源码：

```sh
.venv/bin/python scripts/resolve_history.py de1adf0
```

输出新的完整SHA，可用于`git show`、`git checkout`、`git worktree add`或需要Git提交参数的历史验证脚本。映射只针对本仓库，不修改模型与数据集上游revision。实验run中保存的源码/依赖内容哈希继续有效；当前文档中的历史链接已指向更正后的等价提交。

已有其他clone时，应先保存各自的未提交工作，再按更正后的历史同步或重新clone，避免把旧署名历史合并回main。

## GitHub展示

GitHub根据关联邮箱识别提交归属，本次使用账号对应的ID式隐私邮箱。提交页可先核对作者；Contributors等统计在历史重写后可能需要约24小时刷新，见[GitHub贡献者说明](https://docs.github.com/en/enterprise-cloud@latest/repositories/viewing-activity-and-data-for-your-repository/viewing-a-projects-contributors)。
