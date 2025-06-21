#!/bin/bash
# filepath: /home/sharing/disk1/yanan/LLM-LNS/run.sh

# 定义颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 进度条函数
show_progress() {
    local step=$1
    local total=$2
    local desc=$3
    local percent=$((step * 100 / total))
    local filled=$((step * 50 / total))
    local empty=$((50 - filled))
    
    printf "\r${BLUE}[%02d/%02d]${NC} " $step $total
    printf "${GREEN}"
    for ((i=0; i<filled; i++)); do printf "█"; done
    printf "${NC}"
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf " ${percent}%% - ${desc}"
    
    if [ $step -eq $total ]; then
        printf "\n"
    fi
}

# 总步骤数
TOTAL_STEPS=7

echo -e "${YELLOW}🚀 开始清理 Git 仓库并重新推送...${NC}"
echo "================================"

# 步骤 1: 创建新的干净目录
show_progress 1 $TOTAL_STEPS "创建新的干净目录..."
cd ..
if [ -d "LLM-LNS-clean" ]; then
    echo -e "\n${YELLOW}⚠️  删除已存在的 LLM-LNS-clean...${NC}"
    rm -rf LLM-LNS-clean
fi
mkdir LLM-LNS-clean
sleep 0.5

# 步骤 2: 只复制代码文件（不包括 .git）
show_progress 2 $TOTAL_STEPS "复制代码文件（排除Git历史）..."
cd LLM-LNS
# 使用 rsync 排除 .git 文件夹
rsync -av --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' . ../LLM-LNS-clean/
cd ../LLM-LNS-clean
sleep 1

# 步骤 3: 初始化新的 Git 仓库
show_progress 3 $TOTAL_STEPS "初始化新的 Git 仓库..."
git init >/dev/null 2>&1
sleep 0.5

# 步骤 4: 添加所有文件
show_progress 4 $TOTAL_STEPS "添加文件到 Git..."
git add . >/dev/null 2>&1
sleep 1

# 步骤 5: 提交更改
show_progress 5 $TOTAL_STEPS "提交更改..."
git commit -m "Clean repository without large files" >/dev/null 2>&1
sleep 0.5

# 步骤 6: 配置远程仓库
show_progress 6 $TOTAL_STEPS "配置远程仓库..."
git remote add origin https://github.com/thuiar/LLM-LNS.git >/dev/null 2>&1
git branch -M main >/dev/null 2>&1
sleep 0.5

# 步骤 7: 推送到远程
show_progress 7 $TOTAL_STEPS "准备推送..."
echo -e "\n"

echo -e "${BLUE}📡 开始推送到远程仓库...${NC}"
echo "================================"

# 显示推送进度的函数
push_with_progress() {
    echo -e "${YELLOW}🔄 正在推送，请稍候...${NC}"
    
    # 显示要推送的文件统计
    echo -e "${BLUE}📊 推送统计：${NC}"
    echo "文件数量: $(find . -type f -not -path "./.git/*" | wc -l)"
    echo "仓库大小: $(du -sh .git 2>/dev/null | cut -f1)"
    echo ""
    
    # 使用 proxychains 推送
    if command -v proxychains >/dev/null 2>&1; then
        echo -e "${BLUE}📶 使用代理推送...${NC}"
        proxychains git push --force origin main
    else
        echo -e "${BLUE}📶 直接推送...${NC}"
        git push --force origin main
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ 推送成功！${NC}"
        echo -e "${GREEN}🎉 仓库清理完成！${NC}"
        
        # 显示最终统计
        echo "================================"
        echo -e "${BLUE}📊 最终仓库信息：${NC}"
        echo "文件数量: $(find . -type f -not -path "./.git/*" | wc -l)"
        echo "仓库大小: $(du -sh .git 2>/dev/null | cut -f1)"
        echo "当前目录: $(pwd)"
        
    else
        echo -e "${RED}❌ 推送失败！退出码: $exit_code${NC}"
        echo -e "${YELLOW}💡 可能的解决方案：${NC}"
        echo "1. 检查网络连接"
        echo "2. 检查 GitHub 凭据"
        echo "3. 手动运行: git push --force origin main"
    fi
    
    return $exit_code
}

# 执行推送
push_with_progress

echo "================================"
echo -e "${GREEN}🏁 脚本执行完成！${NC}"