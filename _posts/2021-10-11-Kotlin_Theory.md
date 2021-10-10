# 0.환경설정
### 1) JDK(Java SE)
- Java_Home
- Path <br>

### 2) Android Studio (JetBrains - intellij)
- 안드로이드 sdk - API level 31
- Plugin(sdk tools) - HAXM(Emulator : 하드웨어를 본 딴 소프트웨어)
- Android Virtual Device

## 컴파일 과정
### 1) c언어
source -- 컴파일 --> exe
### 2) java
source -- (a.java) 컴파일 --> Byte code --(a.class) 인터프리터 (jvm)--> running
                         
### 3) kotlin
source -- 컴파일 --> Byte code -- 인터프리터 --> running
APK
- Byte code에서 DEX파일을 만듬 (.class)
- DEX을 받아 러닝 시켜주는 ART

<table>
  <tr>
    <td rowspan="3">내용</td>
    <td>내용</td>
  </tr>
  <tr>
    <td>내용</td>
    <td>내용</td>
  </tr>
  <tr>
    <td>내용</td>
    <td>내용</td>
  </tr>
  <tr>
    <td>내용</td>
    <td>내용</td>
  </tr>
  <tr>
    <td>내용</td>
    <td>내용</td>
  </tr>
  <tr>
    <td>내용</td>
    <td>내용</td>
  </tr>
</table>
                      
---
# 1. 코틀린 개요
### 1) class가 없어도 시작가능, c++과 비슷
- Data Type : Int, Short, Byte, Float => class이기 때문에 대문자
- Value, variable => class 타입의 변수이기 때문에 object가 된다.

### 2) 코틀린은 Strong Type, Week Type 중 Strong Type을 사용한다.

### 3) val, var
- val : 변경 불가 (read only)
- var : 변경 가능
```kotlin
val x : short = 10
var y : Int = 5
```
### 4) scope


### 5) 함수
```kotlin
class Rect {
  var r:Int
  fun areaSize() {
  }
}
// 
var r : Rect = Rect()
```
