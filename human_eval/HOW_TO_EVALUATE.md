# 사람 평가 방법 (diff-description accuracy audit)

## 0. 평가할 파일

```text
메인 평가표 : ./IST_human_eval/out/rater.csv   (80행)
재채점표    : ./IST_human_eval/out/retest.csv  (15행)
정답표(금지): ./IST_human_eval/out/key.csv
```

`key.csv`는 평가 전 열지 말 것. gold/generated, NLI 점수, cell 정보가 들어 있어 blind 평가가 깨진다.

## 1. 평가 질문

각 행은 `(보이는 diff, commit message)` 한 쌍이다.

질문:

> **보이는 diff만 근거로, 이 message가 중요한 변경 내용을 정확하게 설명하는가?**

여기서 보이는 diff는 NLI가 본 diff premise와 맞추기 위해 1,500자로 제한되어 있다. 전체 저장소, 이슈, 테스트 결과, 외부 맥락은 사용하지 않는다.

## 2. 입력할 컬럼

각 행에 아래 두 칸은 필수 입력:

```text
rater_label
rater_reason
```

`rater_notes`는 선택 입력이다.

## 3. `rater_label` 3개

```text
supported
neutral
contradicted
```

- `supported`: message가 보이는 diff의 핵심 변경을 정확하게 설명한다. 모든 세부사항을 다 말할 필요는 없지만, 핵심을 왜곡하거나 diff 밖 주장을 핵심처럼 말하면 안 된다.
- `neutral`: 틀렸다고 하긴 어렵지만, 보이는 diff만으로 정확한 설명이라고 확정하기 어렵다.
- `contradicted`: message가 보이는 diff와 충돌한다. 반대 방향, 잘못된 대상, 잘못된 작업, 다른 변경을 말한다.

## 4. `rater_reason` 선택지

라벨에 맞는 reason 하나를 정확히 입력한다.

### supported

```text
exact_core
partial_but_sufficient
high_level_but_correct
```

- `exact_core`: 핵심 변경을 정확히 잡았다.
- `partial_but_sufficient`: 일부 세부사항은 빠졌지만 중요한 변경 설명으로 충분하다.
- `high_level_but_correct`: 추상적이지만 보이는 diff의 전체 방향을 정확히 설명한다.

### neutral

```text
too_abstract
missing_key_detail
near_miss
external_context
diff_insufficient
ambiguous_vague
other_neutral
```

- `too_abstract`: "improve", "update", "cleanup", "refactor"처럼 너무 일반적이라 정확히 검증하기 어렵다.
- `missing_key_detail`: 실제 변경 일부는 말하지만 중요한 다른 변경을 빠뜨렸다.
- `near_miss`: 의미가 비슷하지만 정확히는 다른 내용을 말한다.
- `external_context`: 버그 원인, 이슈, 의도, 런타임 현상, 이유 등 diff 밖 맥락을 말한다.
- `diff_insufficient`: 보이는 1,500자 diff만으로는 판단 근거가 부족하다.
- `ambiguous_vague`: message 자체가 모호해서 정확한 판단이 어렵다.
- `other_neutral`: 위에 없는 neutral 사유. `rater_notes`에 짧게 적는다.

### contradicted

```text
opposite_direction
wrong_entity
wrong_action
different_change
other_contradicted
```

- `opposite_direction`: add/remove, enable/disable, increase/decrease 같은 방향이 반대다.
- `wrong_entity`: 파일, 함수, 변수, 모듈, API, feature 대상이 다르다.
- `wrong_action`: 대상은 비슷하지만 실제 작업이 다르다.
- `different_change`: 보이는 diff와 다른 변경을 설명한다.
- `other_contradicted`: 위에 없는 contradiction 사유. `rater_notes`에 짧게 적는다.

## 5. 판정 순서

1. diff를 읽고 중요한 변경을 파악한다.
2. message를 읽는다.
3. "보이는 diff만으로 이 message가 중요한 변경을 정확히 설명한다고 볼 수 있는가?"를 판단한다.
4. `rater_label` 하나를 고른다.
5. 해당 label에 맞는 `rater_reason` 하나를 고른다.

## 6. 분석 기준

Primary metric:

```text
supported-rate = rater_label == supported 비율
```

Secondary metrics:

```text
label distribution = supported / neutral / contradicted 분포
reason distribution = rater_reason별 분포
```

gold/generated 비교는 평가자가 보지 못한 `key.csv`와 결합해서 분석 스크립트가 계산한다.

## 7. 분석 실행

채점 후:

```bash
cd "./IST_human_eval"
python3 analyze_human_eval.py
```

결과:

```text
out/agreement.md
out/summary.csv
out/per_cell.csv
out/label_distribution.csv
out/reason_distribution.csv
```
