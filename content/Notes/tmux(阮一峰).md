已经是2019的文章了  
# 来自
[Tmux 使用教程 - 阮一峰的网络日志](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

# 基础知识

### 会话

**用户与计算机的临时交互称一次会话**  
**每一次命令行交互都是一次会话**  

窗口与进程连在一起，打开窗口会话开始，关闭就结束，进程跟着结束  <- **哪里的进程**？ 
为了解决这个问题，**会话与窗口可以"解绑"**：窗口关闭时，会话并不终止，而是继续运行，等到以后需要的时候，再让会话"绑定"其他窗口。  

>那就是关闭窗口是关闭会话，关闭会话进程结束，进程受控于会话  

### Tmux
**解绑工具**   
- 允许在单个窗口中，同时访问多个会话
- 可以让新窗口"接入"已经存在的会话
- 允许每个会话有多个连接窗口
- 支持窗口任意的垂直和水平拆分。

# 基本用法

### 安装
```shell
# Ubuntu 或 Debian
$ sudo apt-get install tmux

# CentOS 或 Fedora
$ sudo yum install tmux
```

如果将来用mac再说  
安装之后通过tmux进入  
底部有一个状态栏。状态栏的左侧是窗口信息（编号和名称），右侧是系统信息  

_ubuntu22.04.4LTS amd64_  
![[Pasted image 20240626210330.png|600]]

```shell
ctrl+d 或 exit  退出
```

### 前缀键
```
Ctrl+b
```
Tmux 窗口有大量的快捷键。所有快捷键都要通过前缀键唤起。  
默认前缀键是 `ctrl+b`  
`ctrl+b + ?` 显示帮助信息  `q`或`Esc` 退出帮助信息页面  




# 会话管理
### 新建会话

会话编号从0开始  
一般给会话起名  
```shell
tmux new -s <session-name>
```
创建后进入窗口  
### 分离会话

将当前会话与窗口分离  
在tmux窗口中   
```shell
tmux detach
Ctrl+b d
```
> 这个前缀键是先按，然后再选快捷键  

分离后退出窗口  
![[Pasted image 20240626211808.png]]

查看所有tmux会话  
```shell
tmux ls
# or
tmux list-session
```

### 接入会话
```shell
tmux attach
```

用编号  
```shell
tmux attach -t 0
```
用名字  
```shell
tmux attach -t session_name
```

### 杀死会话
```shell
tmux kill-session
```

```shell
tmux kill-session -t 0
tmux kill-session -t session_name
```

### 切换会话
```shell
tmux switch
```

```shell
tmux switch -t 0
tmux switch -t session_name
```

### 重命名会话
```shell
tmux rename_session -t old_name new_name
```

### 会话快捷键

```shell
Ctrl+b d 分离当前的会话
Ctrl+b s 列出所有会话
Ctrl+b $ 重命名当前会话
```

### 最简操作流程
```
新建会话tmux new -s my_session。
在 Tmux 窗口运行所需的程序。
按下快捷键Ctrl+b d将会话分离。
下次使用时，重新连接到会话tmux attach-session -t my_session。
```

# 窗格操作
Tmux可以将窗口分成多个窗格(pane)，每个窗格运行不同命令  

在tmux窗口中

### 划分窗格
```shell
tmux split-window
```
- 默认上下
- `tmux split-window -h`  左右

### 移动光标
在窗格间移动光标  
```shell
tmux select-pane
```

```shell
tmux select-pane -U  # 上方
tmux select-pane -D  # 下方
tmux select-pane -L  # 左边
tmux select-pane -R  # 右边
```

### 交换窗格位置
交换窗格本身的位置
```shell
tmux swap-pane
```

```shell
tmux swap-pane -U # 向上
tmux swap-pane -D # 向下
```

### 快捷键
```
Ctrl+b %：划分左右两个窗格。

Ctrl+b "：划分上下两个窗格。

Ctrl+b <arrow key>：光标切换到其他窗格。<arrow key>是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键↓。

Ctrl+b ;：光标切换到上一个窗格。

Ctrl+b o：光标切换到下一个窗格。

Ctrl+b {：当前窗格与上一个窗格交换位置。

Ctrl+b }：当前窗格与下一个窗格交换位置。

Ctrl+b Ctrl+o：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。

Ctrl+b Alt+o：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。

Ctrl+b x：关闭当前窗格。

Ctrl+b !：将当前窗格拆分为一个独立窗口。

Ctrl+b z：当前窗格全屏显示，再使用一次会变回原来大小。

Ctrl+b Ctrl+<arrow key>：按箭头方向调整窗格大小。

Ctrl+b q：显示窗格编号。
```

# 窗口管理

### 创建新窗口
```shell
tmux new-window
```

```shell
tmux new-window -n window_name
```

### 切换窗口
```shell
tmux select-window -t
```

```shell
tmux select-window -t numbe or name
```

### 重命名窗口
```shell
tmux rename-window new_name
```

### 窗口操作的快捷键

```
Ctrl+b c：创建一个新窗口，状态栏会显示多个窗口的信息。

Ctrl+b p：切换到上一个窗口（按照状态栏上的顺序）。

Ctrl+b n：切换到下一个窗口。

Ctrl+b <number>：切换到指定编号的窗口，其中的<number>是状态栏上的窗口编号。

Ctrl+b w：从列表中选择窗口。

Ctrl+b ,：窗口重命名。
```


# 其他命令

```shell
tmux list-keys  # 列出所有快捷键，及其对应的 Tmux 命令

tmux list-commnds  # 列出所有 Tmux 命令及其参数

tmux info # 列出当前所有 Tmux 会话的信息

tmux source-file ~/.tmux.conf # 重新加载当前的 Tmux 配置
```


# 其他参考
- [GitHub - tmux/tmux: tmux source code](https://github.com/tmux/tmux)
- [Site Unreachable](https://linuxize.com/post/getting-started-with-tmux/)
- [A Quick and Easy Guide to tmux - Ham Vocke](https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)