// Copyright (c) HashiCorp, Inc
// SPDX-License-Identifier: MPL-2.0
import { Construct } from "constructs";
import { App, TerraformStack } from "cdktf";
import { AwsProvider } from "@cdktf/provider-aws/lib/provider";
import { EcsCluster } from "@cdktf/provider-aws/lib/ecs-cluster";
import { EcsTaskDefinition } from "@cdktf/provider-aws/lib/ecs-task-definition";

const region = "us-west-1";
const profile = "forecapp";
const name = "forecapp-api";
const tags = {
  project: "forecapp-api",
};

class MyStack extends TerraformStack {
  constructor(scope: Construct, id: string) {
    super(scope, id);

    new AwsProvider(this, "aws", { region, profile });

    new EcsCluster(this, `ecs-${name}`, {
      name,
      tags,
    });

    new EcsTaskDefinition(this, `ecs-task-${name}`, {
      family: "service",
      containerDefinitions: JSON.stringify([
        {
          name,
          portMappings: [
            {
              containerPort: 80,
              hostPort: 80,
            },
          ],
        },
      ]),
      tags,
    });
  }
}

const app = new App();
new MyStack(app, "forecapp-api");
app.synth();
