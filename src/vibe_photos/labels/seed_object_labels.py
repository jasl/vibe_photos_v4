"""Seed initial object labels and aliases for the M2 label layer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from utils.logging import get_logger
from vibe_photos.config import load_settings
from vibe_photos.db import open_primary_session
from vibe_photos.labels.repository import LabelRepository

LOGGER = get_logger(__name__, extra={"command": "seed_object_labels"})


@dataclass(frozen=True)
class ObjectLabelSpec:
    """Schema for a single object label seed entry."""

    key: str
    display_name: str
    parent: str | None
    aliases_en: Sequence[str] = field(default_factory=tuple)
    aliases_zh: Sequence[str] = field(default_factory=tuple)


OBJECT_LABEL_SPECS: tuple[ObjectLabelSpec, ...] = (
    ObjectLabelSpec(
        key="object",
        display_name="物体",
        parent=None,
        aliases_en=("object", "thing"),
        aliases_zh=("物体",),
    ),
    ObjectLabelSpec(
        key="object.electronics",
        display_name="电子产品",
        parent="object",
        aliases_en=("electronic device", "consumer electronics"),
        aliases_zh=("电子产品", "数码产品"),
    ),
    ObjectLabelSpec(
        key="object.food",
        display_name="食物",
        parent="object",
        aliases_en=("food", "dish", "meal"),
        aliases_zh=("食物", "菜品"),
    ),
    ObjectLabelSpec(
        key="object.drink",
        display_name="饮品",
        parent="object",
        aliases_en=("drink", "beverage"),
        aliases_zh=("饮品", "饮料"),
    ),
    ObjectLabelSpec(
        key="object.document",
        display_name="文档与证件",
        parent="object",
        aliases_en=("document", "paper document", "printed document"),
        aliases_zh=("文档", "纸质文件"),
    ),
    ObjectLabelSpec(
        key="object.stationery",
        display_name="文具与办公用品",
        parent="object",
        aliases_en=("office supplies", "stationery"),
        aliases_zh=("文具", "办公用品"),
    ),
    ObjectLabelSpec(
        key="object.dailystuff",
        display_name="日常物品",
        parent="object",
        aliases_en=("everyday objects", "daily items"),
        aliases_zh=("日常物品", "生活用品"),
    ),
    ObjectLabelSpec(
        key="object.electronics.circuit_board",
        display_name="电路板",
        parent="object.electronics",
        aliases_en=("a circuit board"),
        aliases_zh=("电路板"),
    ),
    ObjectLabelSpec(
        key="object.electronics.expansion_card",
        display_name="硬件",
        parent="object.electronics",
        aliases_en=("an expansion card", "a PCIe board"),
        aliases_zh=("硬件", "内设"),
    ),
    ObjectLabelSpec(
        key="object.electronics.disk",
        display_name="硬盘",
        parent="object.electronics",
        aliases_en=("a hard disk", "a solid state disk", "a disk"),
        aliases_zh=("硬盘", "SSD"),
    ),
    ObjectLabelSpec(
        key="object.electronics.charger",
        display_name="充电器",
        parent="object.electronics",
        aliases_en=("a charger", "a power adapter", "an USB charger"),
        aliases_zh=("充电器"),
    ),
    ObjectLabelSpec(
        key="object.electronics.sbc",
        display_name="单板电脑",
        parent="object.electronics",
        aliases_en=("a single board computer", "a SBC"),
        aliases_zh=("单板电脑", "单片机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.mini_pc",
        display_name="迷你主机",
        parent="object.electronics",
        aliases_en=("a mini-PC", "a compact PC"),
        aliases_zh=("迷你主机", "小主机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.mini_pc.mac_mini",
        display_name="Mac mini",
        parent="object.electronics",
        aliases_en=("an apple mac mini"),
        aliases_zh=("Mac mini"),
    ),
    ObjectLabelSpec(
        key="object.electronics.laptop",
        display_name="笔记本电脑",
        parent="object.electronics",
        aliases_en=("a laptop computer", "a notebook computer", "an open laptop on a desk"),
        aliases_zh=("笔记本电脑", "笔电"),
    ),
    ObjectLabelSpec(
        key="object.electronics.laptop.macbook",
        display_name="MacBook",
        parent="object.electronics.laptop",
        aliases_en=("an apple macbook laptop", "a silver apple macbook on a desk", "a macbook pro laptop"),
        aliases_zh=("MacBook", "苹果笔记本"),
    ),
    ObjectLabelSpec(
        key="object.electronics.phone",
        display_name="智能手机",
        parent="object.electronics",
        aliases_en=("a smartphone", "a mobile phone", "a cell phone in a hand"),
        aliases_zh=("手机", "智能手机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.phone.iphone",
        display_name="iPhone",
        parent="object.electronics.phone",
        aliases_en=("an apple iphone smartphone", "an iphone on a table", "an iphone in someone's hand"),
        aliases_zh=("iPhone", "苹果手机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.tablet",
        display_name="平板电脑",
        parent="object.electronics",
        aliases_en=("a tablet device", "a tablet computer", "a tablet on a desk"),
        aliases_zh=("平板电脑",),
    ),
    ObjectLabelSpec(
        key="object.electronics.tablet.ipad",
        display_name="iPad",
        parent="object.electronics.tablet",
        aliases_en=("an apple ipad tablet", "an ipad with a stylus"),
        aliases_zh=("iPad",),
    ),
    ObjectLabelSpec(
        key="object.electronics.peripheral.keyboard",
        display_name="键盘",
        parent="object.electronics",
        aliases_en=("a computer keyboard", "a mechanical keyboard on a desk", "a wireless keyboard"),
        aliases_zh=("键盘", "机械键盘"),
    ),
    ObjectLabelSpec(
        key="object.electronics.peripheral.mouse",
        display_name="鼠标",
        parent="object.electronics",
        aliases_en=("a computer mouse", "a wireless mouse"),
        aliases_zh=("鼠标",),
    ),
    ObjectLabelSpec(
        key="object.electronics.display.monitor",
        display_name="显示器",
        parent="object.electronics",
        aliases_en=("a computer monitor", "a desktop monitor on a desk", "an external display"),
        aliases_zh=("显示器", "电脑显示器"),
    ),
    ObjectLabelSpec(
        key="object.electronics.tv",
        display_name="电视",
        parent="object.electronics",
        aliases_en=("a television", "a tv screen on a wall"),
        aliases_zh=("电视", "电视机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.peripheral.hub",
        display_name="USB Hub / 集线器",
        parent="object.electronics",
        aliases_en=("a usb hub", "a docking station for laptop"),
        aliases_zh=("扩展坞", "USB 集线器"),
    ),
    ObjectLabelSpec(
        key="object.electronics.peripheral.power_strip",
        display_name="插线板",
        parent="object.electronics",
        aliases_en=("a power strip", "an extension cord with sockets"),
        aliases_zh=("插线板", "排插"),
    ),
    ObjectLabelSpec(
        key="object.electronics.audio.headphones",
        display_name="头戴耳机",
        parent="object.electronics",
        aliases_en=("over ear headphones", "a pair of headphones"),
        aliases_zh=("头戴耳机", "耳机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.audio.earbuds",
        display_name="入耳耳机",
        parent="object.electronics",
        aliases_en=("wireless earbuds", "a pair of in ear earphones"),
        aliases_zh=("入耳式耳机", "耳塞"),
    ),
    ObjectLabelSpec(
        key="object.electronics.audio.earbuds.airpods",
        display_name="AirPods",
        parent="object.electronics.audio.earbuds",
        aliases_en=("apple airpods", "a pair of apple airpods in a charging case", "airpods on a table"),
        aliases_zh=("AirPods", "苹果耳机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.audio.speaker",
        display_name="音箱",
        parent="object.electronics",
        aliases_en=("a bluetooth speaker", "a desktop speaker"),
        aliases_zh=("音箱", "蓝牙音箱"),
    ),
    ObjectLabelSpec(
        key="object.electronics.audio.microphone",
        display_name="麦克风",
        parent="object.electronics",
        aliases_en=("a microphone", "a podcast microphone on a boom arm"),
        aliases_zh=("麦克风", "话筒"),
    ),
    ObjectLabelSpec(
        key="object.electronics.camera",
        display_name="相机",
        parent="object.electronics",
        aliases_en=("a digital camera", "a mirrorless camera", "a camera with a lens"),
        aliases_zh=("相机", "数码相机", "微单"),
    ),
    ObjectLabelSpec(
        key="object.electronics.camera.lens",
        display_name="镜头",
        parent="object.electronics.camera",
        aliases_en=("a camera lens", "an interchangeable lens"),
        aliases_zh=("镜头", "相机镜头"),
    ),
    ObjectLabelSpec(
        key="object.electronics.camera.tripod",
        display_name="三脚架",
        parent="object.electronics",
        aliases_en=("a camera tripod", "a tripod stand"),
        aliases_zh=("三脚架",),
    ),
    ObjectLabelSpec(
        key="object.electronics.storage.external_drive",
        display_name="移动硬盘",
        parent="object.electronics",
        aliases_en=("an external hard drive", "a portable ssd"),
        aliases_zh=("移动硬盘", "移动固态硬盘"),
    ),
    ObjectLabelSpec(
        key="object.electronics.storage.usb_drive",
        display_name="U盘",
        parent="object.electronics",
        aliases_en=("a usb flash drive", "a thumb drive"),
        aliases_zh=("U盘", "优盘"),
    ),
    ObjectLabelSpec(
        key="object.electronics.network.router",
        display_name="路由器",
        parent="object.electronics",
        aliases_en=("a wifi router", "a wireless router with antennas"),
        aliases_zh=("路由器", "无线路由器"),
    ),
    ObjectLabelSpec(
        key="object.electronics.console",
        display_name="游戏主机",
        parent="object.electronics",
        aliases_en=("a game console", "a video game console"),
        aliases_zh=("游戏机", "主机"),
    ),
    ObjectLabelSpec(
        key="object.electronics.console.controller",
        display_name="游戏手柄",
        parent="object.electronics.console",
        aliases_en=("a game controller", "a gamepad"),
        aliases_zh=("游戏手柄", "手柄"),
    ),
    ObjectLabelSpec(
        key="object.food.dish",
        display_name="菜",
        parent="object.food",
        aliases_en=("a plate of food", "a cooked dish on a plate"),
        aliases_zh=("菜", "菜品"),
    ),
    ObjectLabelSpec(
        key="object.food.meal",
        display_name="一餐",
        parent="object.food",
        aliases_en=("a meal on a table", "food on a dining table"),
        aliases_zh=("一餐", "一桌菜"),
    ),
    ObjectLabelSpec(
        key="object.food.pizza",
        display_name="披萨",
        parent="object.food",
        aliases_en=("a pizza", "a whole pizza on a plate", "a slice of pizza on a plate"),
        aliases_zh=("披萨", "比萨"),
    ),
    ObjectLabelSpec(
        key="object.food.burger",
        display_name="汉堡",
        parent="object.food",
        aliases_en=("a hamburger", "a cheeseburger with fries", "a burger in a wrapper"),
        aliases_zh=("汉堡", "汉堡包"),
    ),
    ObjectLabelSpec(
        key="object.food.fries",
        display_name="薯条",
        parent="object.food",
        aliases_en=("french fries", "a portion of fries"),
        aliases_zh=("薯条",),
    ),
    ObjectLabelSpec(
        key="object.food.pasta",
        display_name="意面",
        parent="object.food",
        aliases_en=("a plate of pasta", "spaghetti with sauce"),
        aliases_zh=("意面", "意大利面"),
    ),
    ObjectLabelSpec(
        key="object.food.sandwich",
        display_name="三明治",
        parent="object.food",
        aliases_en=("a sandwich", "a club sandwich"),
        aliases_zh=("三明治",),
    ),
    ObjectLabelSpec(
        key="object.food.salad",
        display_name="沙拉",
        parent="object.food",
        aliases_en=("a bowl of salad", "a fresh salad"),
        aliases_zh=("沙拉", "色拉"),
    ),
    ObjectLabelSpec(
        key="object.food.rice_bowl",
        display_name="盖饭",
        parent="object.food",
        aliases_en=("a rice bowl with toppings", "a bowl of rice with meat and vegetables"),
        aliases_zh=("盖饭", "盖浇饭"),
    ),
    ObjectLabelSpec(
        key="object.food.noodles",
        display_name="面条",
        parent="object.food",
        aliases_en=("a bowl of noodles", "a bowl of ramen"),
        aliases_zh=("面", "面条", "拉面"),
    ),
    ObjectLabelSpec(
        key="object.food.dumplings",
        display_name="饺子",
        parent="object.food",
        aliases_en=("a plate of dumplings", "boiled dumplings on a plate"),
        aliases_zh=("饺子", "水饺"),
    ),
    ObjectLabelSpec(
        key="object.food.hotpot",
        display_name="火锅",
        parent="object.food",
        aliases_en=("a hotpot", "a hot pot with soup and ingredients"),
        aliases_zh=("火锅",),
    ),
    ObjectLabelSpec(
        key="object.food.sushi",
        display_name="寿司",
        parent="object.food",
        aliases_en=("a sushi plate", "assorted sushi"),
        aliases_zh=("寿司",),
    ),
    ObjectLabelSpec(
        key="object.food.fruit",
        display_name="水果",
        parent="object.food",
        aliases_en=("a plate of fruit", "fresh fruit on a plate"),
        aliases_zh=("水果",),
    ),
    ObjectLabelSpec(
        key="object.food.dessert",
        display_name="甜品",
        parent="object.food",
        aliases_en=("a dessert", "a sweet dessert on a plate"),
        aliases_zh=("甜品", "甜点"),
    ),
    ObjectLabelSpec(
        key="object.food.cake",
        display_name="蛋糕",
        parent="object.food.dessert",
        aliases_en=("a slice of cake", "a birthday cake"),
        aliases_zh=("蛋糕",),
    ),
    ObjectLabelSpec(
        key="object.food.ice_cream",
        display_name="冰淇淋",
        parent="object.food.dessert",
        aliases_en=("an ice cream cone", "ice cream in a cup"),
        aliases_zh=("冰淇淋", "雪糕"),
    ),
    ObjectLabelSpec(
        key="object.food.cookie",
        display_name="饼干",
        parent="object.food.dessert",
        aliases_en=("cookies", "a plate of cookies"),
        aliases_zh=("饼干",),
    ),
    ObjectLabelSpec(
        key="object.drink.coffee",
        display_name="咖啡",
        parent="object.drink",
        aliases_en=("a cup of coffee", "a latte in a mug", "a cappuccino with latte art"),
        aliases_zh=("咖啡",),
    ),
    ObjectLabelSpec(
        key="object.drink.tea",
        display_name="茶",
        parent="object.drink",
        aliases_en=("a cup of tea", "a teapot and a tea cup"),
        aliases_zh=("茶", "热茶"),
    ),
    ObjectLabelSpec(
        key="object.drink.milk_tea",
        display_name="奶茶",
        parent="object.drink",
        aliases_en=("a cup of bubble tea", "a cup of milk tea with tapioca pearls"),
        aliases_zh=("奶茶", "珍珠奶茶"),
    ),
    ObjectLabelSpec(
        key="object.drink.juice",
        display_name="果汁",
        parent="object.drink",
        aliases_en=("a glass of juice", "orange juice in a glass"),
        aliases_zh=("果汁",),
    ),
    ObjectLabelSpec(
        key="object.drink.soda",
        display_name="汽水 / 可乐",
        parent="object.drink",
        aliases_en=("a can of soda", "a glass of cola with ice"),
        aliases_zh=("汽水", "可乐"),
    ),
    ObjectLabelSpec(
        key="object.drink.beer",
        display_name="啤酒",
        parent="object.drink",
        aliases_en=("a glass of beer", "a beer bottle and a glass"),
        aliases_zh=("啤酒",),
    ),
    ObjectLabelSpec(
        key="object.drink.wine",
        display_name="红酒",
        parent="object.drink",
        aliases_en=("a glass of red wine", "a bottle of wine and a wine glass"),
        aliases_zh=("红酒", "葡萄酒"),
    ),
    ObjectLabelSpec(
        key="object.drink.water_bottle",
        display_name="水瓶 / 矿泉水",
        parent="object.drink",
        aliases_en=("a plastic water bottle", "a reusable water bottle"),
        aliases_zh=("矿泉水", "水瓶"),
    ),
    ObjectLabelSpec(
        key="object.document.paper",
        display_name="纸质文档",
        parent="object.document",
        aliases_en=("a printed document", "a sheet of paper with text"),
        aliases_zh=("纸质文档", "打印文件"),
    ),
    ObjectLabelSpec(
        key="object.document.book",
        display_name="书",
        parent="object.document",
        aliases_en=("a book", "an open book"),
        aliases_zh=("书", "书本"),
    ),
    ObjectLabelSpec(
        key="object.document.notebook",
        display_name="纸质笔记本",
        parent="object.document",
        aliases_en=("a paper notebook", "a notebook with handwritten notes"),
        aliases_zh=("笔记本", "本子"),
    ),
    ObjectLabelSpec(
        key="object.document.id_card",
        display_name="证件卡",
        parent="object.document",
        aliases_en=("an id card", "a plastic identity card"),
        aliases_zh=("身份证", "证件卡"),
    ),
    ObjectLabelSpec(
        key="object.document.bank_card",
        display_name="银行卡 / 信用卡",
        parent="object.document",
        aliases_en=("a credit card", "a bank card"),
        aliases_zh=("银行卡", "信用卡"),
    ),
    ObjectLabelSpec(
        key="object.document.receipt",
        display_name="小票 / 收据",
        parent="object.document",
        aliases_en=("a receipt", "a printed receipt on paper"),
        aliases_zh=("小票", "收据"),
    ),
    ObjectLabelSpec(
        key="object.stationery.pen",
        display_name="笔",
        parent="object.stationery",
        aliases_en=("a pen", "a ballpoint pen"),
        aliases_zh=("笔", "圆珠笔"),
    ),
    ObjectLabelSpec(
        key="object.stationery.scissors",
        display_name="剪刀",
        parent="object.stationery",
        aliases_en=("a pair of scissors",),
        aliases_zh=("剪刀",),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.mug",
        display_name="杯子 / 马克杯",
        parent="object.dailystuff",
        aliases_en=("a ceramic mug", "a coffee mug on a desk"),
        aliases_zh=("杯子", "马克杯"),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.bottle",
        display_name="水杯 / 保温杯",
        parent="object.dailystuff",
        aliases_en=("a thermos bottle", "a reusable bottle"),
        aliases_zh=("保温杯", "水杯"),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.backpack",
        display_name="背包",
        parent="object.dailystuff",
        aliases_en=("a backpack", "a travel backpack"),
        aliases_zh=("背包", "双肩包"),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.suitcase",
        display_name="行李箱",
        parent="object.dailystuff",
        aliases_en=("a suitcase", "a rolling suitcase"),
        aliases_zh=("行李箱",),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.chair",
        display_name="椅子",
        parent="object.dailystuff",
        aliases_en=("a chair", "an office chair"),
        aliases_zh=("椅子",),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.desk",
        display_name="桌子 / 书桌",
        parent="object.dailystuff",
        aliases_en=("a desk", "a wooden desk"),
        aliases_zh=("桌子", "书桌"),
    ),
    ObjectLabelSpec(
        key="object.dailystuff.lamp",
        display_name="台灯 / 灯",
        parent="object.dailystuff",
        aliases_en=("a desk lamp", "a table lamp"),
        aliases_zh=("台灯", "灯"),
    ),
)


def seed_object_labels(session: Session) -> None:
    """Insert or update the initial set of object labels and aliases."""

    repo = LabelRepository(session)
    for spec in OBJECT_LABEL_SPECS:
        label = repo.get_or_create_label(
            key=spec.key,
            level="object",
            display_name=spec.display_name,
            parent_key=spec.parent,
        )
        repo.ensure_aliases(label, spec.aliases_en, language="en")
        repo.ensure_aliases(label, spec.aliases_zh, language="zh")

    session.commit()
    LOGGER.info("seed_object_labels_complete", extra={"count": len(OBJECT_LABEL_SPECS)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed object labels and aliases into the data DB.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Primary PostgreSQL database URL. Defaults to databases.primary_url in settings.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    target = args.db or settings.databases.primary_url
    with open_primary_session(target) as session:
        seed_object_labels(session)


if __name__ == "__main__":
    main()
